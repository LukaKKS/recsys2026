"""
FALCON v2.1: Field-Aware Lightweight COmpression with pressure-derived policy

v2 → v2.1 변경 사항:
    v2 문제: within-field hot/cold 분리
             → M_f를 0.1/0.9로 쪼갬
             → SMED cold-start 구간에 모든 feature가 M_cold(90%)에 집중
             → 충돌 폭발 → 성능 저하

    v2.1 해결: hot/cold 분리 제거, 단일 HashEmbedding 사용
               → M_f 전체를 한 테이블로 사용
               → SMED cold-start 문제 없음
               → theory-driven k로 충돌 최소화

3개 모듈:
    FieldWiseSMED        : field별 독립 빈도 스케치 (비동기 유지, 나중 확장용)
    FieldBudgetAllocator : pressure 기반 field-aware 메모리 배분 (단일 M_f)
    AdaptiveKSelector    : P* 기반 이론적 k 자동 결정 (k_min=6으로 안전 마진 확보)
    FALCONEncoder        : 위 3개 모듈을 통합한 전체 파이프라인
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from typing import List, Dict, Optional, Tuple, Set

from datasketches import frequent_items_sketch, frequent_items_error_type
from hash_embedding import HashEmbedding


# ---------------------------------------------------------------------------
# 1. FieldWiseSMED  (비동기 유지, 나중 확장용)
# ---------------------------------------------------------------------------

class FieldWiseSMED:
    """field마다 독립적인 SMED sketch (비동기 업데이트).

    v2.1에서는 encode() 라우팅에 사용하지 않음.
    코드 구조는 유지 — 나중에 k 동적 조정, 전환 감지 등에 활용 가능.

    K_f = K_base × max(1, int(log2(card_f)))
    ThreadPoolExecutor(max_workers=4): GPU 학습과 CPU sketch 업데이트 병렬 수행
    """

    def __init__(self, field_cardinalities: List[int], K_base: int = 256):
        self.n_fields = len(field_cardinalities)
        self.cardinalities = field_cardinalities
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.K_per_field: List[int] = []
        self.sketches: List[frequent_items_sketch] = []
        for card in field_cardinalities:
            log2_card = max(1, int(math.log2(max(card, 2))))
            K_f = K_base * log2_card
            self.K_per_field.append(K_f)
            lg_max_k = max(4, int(math.log2(max(K_f, 16))))
            self.sketches.append(frequent_items_sketch(lg_max_k))

    # ------------------------------------------------------------------
    def update_async(self, batch_cat_features: List[torch.Tensor]) -> list:
        """현재 배치를 배경 스레드에서 비동기로 sketch 업데이트.
        GPU→CPU 복사는 메인 스레드에서 수행 (CUDA 안전).
        """
        cpu_copies = [v.detach().cpu() for v in batch_cat_features]

        def _update_field(field_idx: int, cpu_vals: torch.Tensor) -> None:
            sketch = self.sketches[field_idx]
            for v in cpu_vals.numpy():
                sketch.update(int(v))

        futures = [
            self.executor.submit(_update_field, i, cpu_copies[i])
            for i in range(self.n_fields)
        ]
        return futures

    # ------------------------------------------------------------------
    def get_frequent(self, field_idx: int) -> Set[int]:
        """해당 field의 고빈도 feature 집합 반환."""
        items = self.sketches[field_idx].get_frequent_items(
            err_type=frequent_items_error_type.NO_FALSE_POSITIVES,
            threshold=0,
        )
        return {item[0] for item in items}


# ---------------------------------------------------------------------------
# 2. FieldBudgetAllocator  (단일 M_f, hot/cold 분리 없음)
# ---------------------------------------------------------------------------

class FieldBudgetAllocator:
    """카디널리티 pressure 기반 field-aware 메모리 배분.

    pressure_f = log(card_f) / Σ log(card_i)
    M_f = M_total × pressure_f   (min 1, 카디널리티 상한 적용)

    v2와의 차이: M_hot / M_cold 분리 없음.
    M_f 전체를 단일 HashEmbedding에 사용 → SMED cold-start 문제 해소.

    음수 budget 방지 (CR=10000):
        1단계: 비례 배분 (floor at 1)
        2단계: M_total 초과 시 비율 축소 (floor at 1)
        3단계: 카디널리티 상한 적용
    """

    def __init__(
        self,
        M_total: int,
        field_cardinalities: List[int],
        min_rows: int = 1,
    ):
        self.M_total = M_total
        self.field_cardinalities = field_cardinalities
        self.n_fields = len(field_cardinalities)

        # pressure 계산 (카디널리티 기반, 학습 중 고정)
        log_cards = [math.log(max(c, 2)) for c in field_cardinalities]
        total_log = sum(log_cards)
        self.pressures: List[float] = [lc / total_log for lc in log_cards]

        # 1단계: 비례 배분
        raw = [max(min_rows, int(M_total * p)) for p in self.pressures]

        # 2단계: M_total 초과 시 비율 축소
        total = sum(raw)
        if total > M_total and total > 0:
            scale = M_total / total
            raw = [max(min_rows, int(r * scale)) for r in raw]

        # 3단계: 카디널리티 상한 적용 (여유 field는 card로 고정)
        self.M_f: List[int] = [
            min(card, budget)
            for card, budget in zip(field_cardinalities, raw)
        ]

        # 압축 필요 여부 (card > M_f → 해싱 필요)
        self.compressed_mask: List[bool] = [
            card > m for card, m in zip(field_cardinalities, self.M_f)
        ]


# ---------------------------------------------------------------------------
# 3. AdaptiveKSelector  (P*-based theory-driven, k_min=6)
# ---------------------------------------------------------------------------

class AdaptiveKSelector:
    """목표 충돌률 P*를 만족하는 최솟값 k를 이론적으로 역산.

    LEAF 논문 Section 3.2 수식:
        P_col = 1 - exp(-|X_f| / M_f^k) ≤ P*

    역산:
        k_f = ceil(log(|X_f| / (-log(1-P*))) / log(M_f))

    v2.1 변경:
        k_min = 6  (v2의 2 → 6으로 상향, 안전 마진 확보)
        이유: SMED 라우팅 없이 단일 테이블 사용 → 충돌 여유 필요

    LEAF k=12 대비:
        큰 field: k=8~12 (이론 최솟값 기반)
        작은 field: k=6 (min 보장)
        → field별 최적화, 낭비 없음
    """

    def __init__(self, P_star: float = 0.01, k_min: int = 6, k_max: int = 32):
        self.P_star = P_star
        self.k_min = k_min
        self.k_max = k_max

    # ------------------------------------------------------------------
    def compute_k(self, n_features: int, M: int) -> int:
        """이론적 최솟값 k 계산.

        Returns
        -------
        k = 1  : M >= n_features (raw-pass 신호)
        k ∈ [k_min, k_max] : 압축 field
        """
        if M <= 0:
            return self.k_max
        if M >= n_features:
            return 1        # raw-pass 신호 (FALCONEncoder에서 사용 안 됨)
        if n_features <= 0 or M <= 1:
            return self.k_max

        try:
            # k ≥ ceil(log(n / (-log(1-P*))) / log(M))
            denom = -math.log(1.0 - self.P_star)   # P* < 1 이므로 양수
            target = math.log(max(n_features / denom, 1.0))
            k = math.ceil(target / math.log(M))
            return max(self.k_min, min(self.k_max, k))
        except (ValueError, ZeroDivisionError):
            return self.k_max


# ---------------------------------------------------------------------------
# 4. FALCONEncoder  (v2.1: 단일 테이블)
# ---------------------------------------------------------------------------

class FALCONEncoder:
    """FALCON v2.1 전체 파이프라인.

    초기화 시 (1회):
        1. FieldWiseSMED  — 비동기 업데이트 준비 (나중 확장용)
        2. FieldBudgetAllocator — pressure 기반 단일 M_f 결정
        3. AdaptiveKSelector (k_min=6) — P* 이론값으로 k_f 계산
        4. 압축 field별 단일 HashEmbedding 생성

    encode() 시 (매 배치):
        1. SMED 비동기 업데이트 시작 (GPU 학습과 병렬, 나중 확장용)
        2. field 루프:
            - raw-pass  : raw_idx 직접 통과 (1 index/sample)
            - compress  : HashEmbedding k_f 해시 → k indices/sample
                          flat_indices = [batch × k], offsets = [0, k, 2k, ...]
    """

    def __init__(
        self,
        field_cardinalities: List[int],
        M_total: int,
        arch_sparse_feature_size: int = 16,
        P_star: float = 0.01,
        K_base: int = 256,
        device: Optional[torch.device] = None,
    ):
        self.field_cardinalities = field_cardinalities
        self.n_fields = len(field_cardinalities)
        self.M_total = M_total
        self.arch_sparse_feature_size = arch_sparse_feature_size
        self.device = device or torch.device("cpu")

        # ── 3개 모듈 초기화 ──────────────────────────────────────────────
        self.field_smed = FieldWiseSMED(field_cardinalities, K_base)

        self.budget = FieldBudgetAllocator(M_total, field_cardinalities)

        self.k_selector = AdaptiveKSelector(P_star, k_min=6)

        # ── field별 k 계산 ─────────────────────────────────────────────
        self.k_f: List[int] = [
            self.k_selector.compute_k(card, self.budget.M_f[i])
            for i, card in enumerate(field_cardinalities)
        ]

        # ── 단일 HashEmbedding 생성 ────────────────────────────────────
        self._build_embeddings()

        # 비동기 SMED futures
        self._pending_futures: list = []

    # ------------------------------------------------------------------
    def _build_embeddings(self) -> None:
        """압축 field에만 HashEmbedding 생성 (단일 테이블)."""
        self.embeddings: Dict[int, HashEmbedding] = {}

        for i, card in enumerate(self.field_cardinalities):
            if not self.budget.compressed_mask[i]:
                continue  # raw-pass: HashEmbedding 불필요
            self.embeddings[i] = HashEmbedding(
                card,
                self.arch_sparse_feature_size,
                num_buckets=self.budget.M_f[i],
                num_hashes=self.k_f[i],
                append_weight=False,
                offsets=0,
                device=self.device,
            )

    # ------------------------------------------------------------------
    def encode(
        self,
        batch_cat_features: List[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """전체 배치 인코딩.

        압축 field:
            hashed = [k_f, batch_size]
            flat_indices = hashed.T.reshape(-1)  → [batch_size × k_f]
            offsets = [0, k_f, 2·k_f, ...]       → [batch_size]
        raw-pass field:
            indices = raw_idx                     → [batch_size]
            offsets = [0, 1, 2, ...]              → [batch_size]

        Returns
        -------
        encoded_indices, encoded_offsets
        """
        dev = device or self.device
        batch_size = batch_cat_features[0].shape[0]

        # SMED 비동기 업데이트 (GPU 학습과 병렬, v2.1에서는 라우팅에 미사용)
        # 이전 futures는 GC에 맡기고 새로 발행 (완료 대기 불필요)
        self._pending_futures = self.field_smed.update_async(batch_cat_features)

        # raw-pass용 공용 offsets (1 index/sample)
        simple_offsets = torch.arange(batch_size, dtype=torch.long, device=dev)

        encoded_indices: List[torch.Tensor] = []
        encoded_offsets: List[torch.Tensor] = []

        for field_idx, raw_idx in enumerate(batch_cat_features):
            raw_idx = raw_idx.to(dev)

            if not self.budget.compressed_mask[field_idx]:
                # raw-pass: 해싱 없이 직접 통과
                encoded_indices.append(raw_idx)
                encoded_offsets.append(simple_offsets)
            else:
                # 단일 테이블 해싱: k_f 해시 함수 모두 사용
                k = self.k_f[field_idx]
                he = self.embeddings[field_idx]
                hashed = he.get_hash_embedding_tensors(
                    raw_idx.to(self.device)
                )  # [k, batch_size]

                # EmbeddingBag: k indices/sample → 평균 임베딩
                flat_indices = hashed.T.reshape(-1).to(dev)   # [batch × k]
                offsets = torch.arange(
                    0, batch_size * k, k, dtype=torch.long, device=dev
                )                                              # [0, k, 2k, ...]
                encoded_indices.append(flat_indices)
                encoded_offsets.append(offsets)

        return encoded_indices, encoded_offsets

    # ------------------------------------------------------------------
    def get_field_ln_emb(self) -> List[int]:
        """DLRM 모델 초기화용: field별 embedding table 크기 반환."""
        return list(self.budget.M_f)

    # ------------------------------------------------------------------
    def print_field_stats(self) -> None:
        """field별 통계 출력."""
        n_compress = sum(self.budget.compressed_mask)
        print(
            f"[FALCON] 압축 대상 field: {n_compress}/{self.n_fields}개 "
            f"(나머지 {self.n_fields - n_compress}개는 raw-pass)"
        )
        print(
            f"[FALCON] 전체 budget 배분 완료 "
            f"(M_total={self.M_total:,}, fields={self.n_fields})"
        )
        print(f"[FALCON] P* = {self.k_selector.P_star}  (목표 충돌률)")

        for i, card in enumerate(self.field_cardinalities):
            M = self.budget.M_f[i]
            k = self.k_f[i]
            mode = "compress" if self.budget.compressed_mask[i] else "raw-pass"

            # 실제 P_col 계산 (raw-pass field는 압축 없음 → P_col = 0)
            if not self.budget.compressed_mask[i]:
                p_col = 0.0
            else:
                try:
                    exp_arg = -card / max(M ** k, 1e-300)
                    p_col = 1.0 - math.exp(max(exp_arg, -500.0))
                except (OverflowError, ZeroDivisionError):
                    p_col = 0.0 if M ** k > card else 1.0

            print(
                f"[FALCON] Field {i:2d}: "
                f"card={card:>12,}, "
                f"k={k:>2}, "
                f"M={M:>8,}, "
                f"pressure={self.budget.pressures[i]:.4f}, "
                f"P_col≈{p_col:.6f}, "
                f"mode={mode}"
            )
