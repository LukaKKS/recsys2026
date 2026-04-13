"""
FALCON v2: Field-Aware Lightweight COmpression with pressure-derived policy

3개 모듈로 구성된 field-aware 임베딩 압축 프레임워크:

    FieldWiseSMED        : field별 독립 빈도 스케치 (비동기 업데이트)
                           K_f = K_base × log2(card_f) — field 크기에 비례한 sketch 용량
                           ThreadPoolExecutor로 GPU 학습과 CPU sketch 업데이트 병렬 수행

    FieldBudgetAllocator : pressure 기반 field-aware 메모리 배분
                           pressure_f = log(card_f) / Σ log(card_i)
                           M_f = M_total × pressure_f, 각 field 내에서 hot/cold 분리

    AdaptiveKSelector    : P* 기반 이론적 k 자동 결정
                           모든 field가 동일한 충돌률 P*를 갖도록 k_f를 역산
                           P_col = 1 - exp(-|X_f| / M_f^k) ≤ P*

    FALCONEncoder        : 위 3개 모듈을 통합한 전체 파이프라인
                           압축 field: hot/cold 분리 인코딩 (1-index-per-sample)
                           여유 field: raw indices 직접 통과 (해싱 스킵)

학습 흐름 (1배치 lag 설계):
    배치 N: SMED 결과(N-1 기반) 읽기 → 라우팅 → 인코딩 → SMED(N) 비동기 업데이트 시작
    배치 N+1: SMED(N) 완료 대기 → SMED 결과(N 기반) 읽기 → ...
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
# 1. FieldWiseSMED
# ---------------------------------------------------------------------------

class FieldWiseSMED:
    """field마다 독립적인 SMED sketch (비동기 업데이트).

    LEAF의 단일 전역 sketch 문제:
        - 대형 field가 카운터를 독점 → 소형 field 빈도 추정 불정확
        - 모든 field에 동일한 K 사용 → 대형 field 정밀도 부족

    FALCON 해결책:
        - field별 K_f = K_base × max(1, int(log2(card_f)))
        - ThreadPoolExecutor(max_workers=4)로 GPU 학습과 병렬 업데이트
        - 학습 속도에 영향 없이 SMED 최신 상태 유지 (1배치 lag)
    """

    def __init__(self, field_cardinalities: List[int], K_base: int = 256):
        self.n_fields = len(field_cardinalities)
        self.cardinalities = field_cardinalities
        self.executor = ThreadPoolExecutor(max_workers=4)

        # field별 K_f 계산 및 sketch 생성
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

        GPU→CPU 복사는 메인 스레드에서 선행 (CUDA 스레드 안전 보장).
        sketch 업데이트(CPU 연산)만 배경 스레드로 위임.
        """
        # GPU→CPU 복사는 메인 스레드에서 수행
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
        """해당 field의 고빈도 feature 집합 반환 (이전 배치 SMED 기준)."""
        items = self.sketches[field_idx].get_frequent_items(
            err_type=frequent_items_error_type.NO_FALSE_POSITIVES,
            threshold=0,
        )
        return {item[0] for item in items}


# ---------------------------------------------------------------------------
# 2. FieldBudgetAllocator
# ---------------------------------------------------------------------------

class FieldBudgetAllocator:
    """카디널리티 pressure 기반 field-aware 메모리 배분.

    pressure_f = log(card_f) / Σ log(card_i)
    M_f = M_total × pressure_f   (min 1, 카디널리티 상한 적용)

    각 field 내부 hot/cold 분리:
        M_cold_f = floor(M_f × long_tail_ratio)
        M_hot_f  = M_f - M_cold_f   (M_hot + M_cold = M_f 보장)

    음수 budget 방지 (CR=10000 등 고압축 시):
        1단계: 비례 배분 (floor at 1)
        2단계: M_total 초과 시 비율로 축소
        3단계: 카디널리티 상한 적용
    """

    def __init__(
        self,
        M_total: int,
        field_cardinalities: List[int],
        long_tail_ratio: float = 0.9,
        min_rows: int = 1,
    ):
        self.M_total = M_total
        self.field_cardinalities = field_cardinalities
        self.n_fields = len(field_cardinalities)
        self.long_tail_ratio = long_tail_ratio

        # pressure 계산 (카디널리티 기반)
        log_cards = [math.log(max(c, 2)) for c in field_cardinalities]
        total_log = sum(log_cards)
        self.pressures: List[float] = [lc / total_log for lc in log_cards]

        # 1단계: 비례 배분 (floor at 1)
        raw = [max(min_rows, int(M_total * p)) for p in self.pressures]

        # 2단계: M_total 초과 시 비율로 축소 (음수 방지)
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

        # hot/cold 분리 (M_hot + M_cold = M_f 정확히 보장)
        self.M_cold: List[int] = [
            max(1, int(m * long_tail_ratio)) for m in self.M_f
        ]
        self.M_hot: List[int] = [
            max(1, m - mc) for m, mc in zip(self.M_f, self.M_cold)
        ]


# ---------------------------------------------------------------------------
# 3. AdaptiveKSelector (P*-based theory-driven)
# ---------------------------------------------------------------------------

class AdaptiveKSelector:
    """목표 충돌률 P*를 만족하는 최소 k를 이론적으로 역산.

    LEAF 논문 Section 3.2 수식:
        P_col = 1 - exp(-|X_f| / M_f^k) ≤ P*

    역산 과정:
        exp(-|X_f| / M_f^k) ≥ 1 - P*
        -|X_f| / M_f^k ≥ log(1 - P*)
        M_f^k ≥ |X_f| / (-log(1 - P*))
        k ≥ log(|X_f| / (-log(1-P*))) / log(M_f)

        → k_f = ceil(log(|X_f| / (-log(1-P*))) / log(M_f))

    효과:
        - 모든 field가 P_col ≤ P* 를 만족 (균등한 충돌률)
        - 작은 field: k 작게 (낭비 없음)
        - 큰 field: k 크게 (충돌 방지)
        - M >= |X_f|: 압축 불필요 → raw-pass

    v1(log-load 기반)과의 차이:
        v1: P_col = max(0, log(card/M)) / k  (경험적)
        v2: LEAF 논문 수식으로 이론적 역산 → 리뷰어 설득력 높음
    """

    def __init__(self, P_star: float = 0.01, k_min: int = 2, k_max: int = 32):
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
            return 1  # raw-pass 신호 (EmbeddingBag에서는 사용 안 됨)
        if n_features <= 0 or M <= 1:
            return self.k_max

        try:
            # P* 기반 이론적 k
            # k ≥ log(n / (-log(1-P*))) / log(M)
            denom = -math.log(1.0 - self.P_star)   # P_star < 1 이므로 양수
            target = math.log(max(n_features / denom, 1.0))
            k = math.ceil(target / math.log(M))
            return max(self.k_min, min(self.k_max, k))
        except (ValueError, ZeroDivisionError):
            return self.k_max


# ---------------------------------------------------------------------------
# 4. FALCONEncoder
# ---------------------------------------------------------------------------

class FALCONEncoder:
    """FALCON v2 전체 파이프라인.

    초기화 시 (1회):
        1. FieldWiseSMED  — field별 독립 sketch 준비
        2. FieldBudgetAllocator — pressure 기반 hot/cold budget 결정
        3. AdaptiveKSelector — P* 이론값으로 k_hot, k_cold 계산
        4. 압축 field별 cold_he / hot_he HashEmbedding 생성
           cold: offsets=0,       bucket [0,        M_cold-1]
           hot:  offsets=M_cold,  bucket [M_cold, M_f-1]
           EmbeddingBag 크기 = M_f = M_cold + M_hot

    encode() 시 (매 배치):
        1. 이전 배치 SMED futures 완료 대기 (읽기 전 쓰기 보장)
        2. SMED get_frequent() 조회 (이전 배치 기준, 1배치 lag)
        3. 현재 배치 SMED 비동기 업데이트 시작 (GPU 학습과 병렬)
        4. field 루프:
            - raw-pass  : raw_idx 직접 통과
            - SMED 미학습: cold_he 첫 번째 해시 사용
            - SMED 학습됨: hot/cold 분리 인코딩 (1-index-per-sample)
    """

    def __init__(
        self,
        field_cardinalities: List[int],
        M_total: int,
        arch_sparse_feature_size: int = 16,
        long_tail_ratio: float = 0.9,
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
        self.budget = FieldBudgetAllocator(
            M_total, field_cardinalities, long_tail_ratio
        )
        self.k_selector = AdaptiveKSelector(P_star)

        # ── k 계산 (압축 field에만 의미 있음) ────────────────────────────
        # 초기 추정: 고빈도 = 상위 (1-long_tail_ratio), 저빈도 = 나머지
        self.k_hot: List[int] = []
        self.k_cold: List[int] = []
        for i, card in enumerate(field_cardinalities):
            if not self.budget.compressed_mask[i]:
                self.k_hot.append(1)
                self.k_cold.append(1)
            else:
                n_hot_est = max(1, int(card * (1.0 - long_tail_ratio)))
                n_cold_est = max(1, card - n_hot_est)
                self.k_hot.append(
                    self.k_selector.compute_k(n_hot_est, self.budget.M_hot[i])
                )
                self.k_cold.append(
                    self.k_selector.compute_k(n_cold_est, self.budget.M_cold[i])
                )

        # ── HashEmbedding 생성 (압축 field에만) ──────────────────────────
        # cold_he: offsets=0        → bucket [0, M_cold-1]
        # hot_he:  offsets=M_cold   → bucket [M_cold, M_f-1]
        # EmbeddingBag 크기 = M_f = M_cold + M_hot
        self.cold_he: Dict[int, HashEmbedding] = {}
        self.hot_he: Dict[int, HashEmbedding] = {}

        n_compressed = sum(self.budget.compressed_mask)
        print(
            f"[FALCON] 압축 대상 field: {n_compressed}/{self.n_fields}개 "
            f"(나머지 {self.n_fields - n_compressed}개는 raw-pass)"
        )

        for i, card in enumerate(field_cardinalities):
            if not self.budget.compressed_mask[i]:
                continue
            M_cold_i = self.budget.M_cold[i]
            M_hot_i = self.budget.M_hot[i]

            self.cold_he[i] = HashEmbedding(
                card,
                arch_sparse_feature_size,
                num_buckets=M_cold_i,
                num_hashes=self.k_cold[i],
                append_weight=False,
                offsets=0,
                device=self.device,
            )
            self.hot_he[i] = HashEmbedding(
                card,
                arch_sparse_feature_size,
                num_buckets=M_hot_i,
                num_hashes=self.k_hot[i],
                append_weight=False,
                offsets=M_cold_i,       # hot bucket [M_cold, M_f-1]
                device=self.device,
            )

        # 비동기 SMED futures (이전 배치)
        self._pending_futures: list = []

    # ------------------------------------------------------------------
    def encode(
        self,
        batch_cat_features: List[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """전체 배치 인코딩.

        Returns
        -------
        encoded_indices : list[Tensor]
        encoded_offsets : list[Tensor]
        """
        dev = device or self.device
        batch_size = batch_cat_features[0].shape[0]

        # 1. 이전 배치 SMED 완료 대기 (읽기 전 쓰기 완료 보장)
        if self._pending_futures:
            futures_wait(self._pending_futures)
            self._pending_futures = []

        # 2. 현재 배치 SMED 비동기 업데이트 시작 (GPU 학습과 병렬)
        self._pending_futures = self.field_smed.update_async(batch_cat_features)

        # 공용 1-index-per-sample offsets (여유 field 및 split 인코딩 공유)
        simple_offsets = torch.arange(batch_size, dtype=torch.long, device=dev)

        encoded_indices: List[torch.Tensor] = []
        encoded_offsets: List[torch.Tensor] = []

        for field_idx, raw_idx in enumerate(batch_cat_features):
            raw_idx = raw_idx.to(dev)

            if not self.budget.compressed_mask[field_idx]:
                # raw-pass: 해싱 없이 직접 통과
                encoded_indices.append(raw_idx)
                encoded_offsets.append(simple_offsets)
                continue

            # 이전 배치의 SMED 결과로 hot 집합 조회
            frequent = self.field_smed.get_frequent(field_idx)

            if len(frequent) == 0:
                # SMED 미학습 구간 (초반): 전부 cold 해시 사용
                he = self.cold_he[field_idx]
                hashed = he.get_hash_embedding_tensors(raw_idx.to(self.device))
                # [k_cold, batch] → 첫 번째 해시 함수 출력 (1-index-per-sample)
                encoded_indices.append(hashed[0].to(dev))
                encoded_offsets.append(simple_offsets)
            else:
                # hot/cold 분리 인코딩
                flat, offs = self._encode_split(
                    field_idx, raw_idx, frequent, dev
                )
                encoded_indices.append(flat)
                encoded_offsets.append(offs)

        return encoded_indices, encoded_offsets

    # ------------------------------------------------------------------
    def _encode_split(
        self,
        field_idx: int,
        raw_idx: torch.Tensor,
        frequent: Set[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """hot/cold 분리 인코딩 (1-index-per-sample).

        hot 샘플 → hot_he  → bucket [M_cold, M_f-1]
        cold 샘플 → cold_he → bucket [0,      M_cold-1]

        torch.isin을 사용해 GPU에서 벡터화 마스킹 (Python 루프 없음).
        """
        batch_size = raw_idx.shape[0]

        # GPU 벡터화 hot_mask
        frequent_t = torch.tensor(
            list(frequent), dtype=raw_idx.dtype, device=device
        )
        hot_mask = torch.isin(raw_idx, frequent_t)  # [batch_size] bool

        results = torch.zeros(batch_size, dtype=torch.long, device=device)

        hot_idx = raw_idx[hot_mask]
        if hot_idx.numel() > 0:
            hashed = self.hot_he[field_idx].get_hash_embedding_tensors(
                hot_idx.to(self.device)
            )  # [k_hot, n_hot]
            results[hot_mask] = hashed[0].to(device)   # 첫 번째 해시 사용

        cold_idx = raw_idx[~hot_mask]
        if cold_idx.numel() > 0:
            hashed = self.cold_he[field_idx].get_hash_embedding_tensors(
                cold_idx.to(self.device)
            )  # [k_cold, n_cold]
            results[~hot_mask] = hashed[0].to(device)

        offsets = torch.arange(batch_size, dtype=torch.long, device=device)
        return results, offsets

    # ------------------------------------------------------------------
    def get_field_ln_emb(self) -> List[int]:
        """DLRM 모델 초기화용: field별 embedding table 크기 반환.

        Returns M_f = M_cold + M_hot for each field.
        여유 field의 경우 M_f = card (자연 크기).
        """
        return list(self.budget.M_f)

    # ------------------------------------------------------------------
    def print_field_stats(self) -> None:
        """field별 통계 출력."""
        print(
            f"[FALCON] 전체 budget 배분 완료 "
            f"(M_total={self.M_total}, fields={self.n_fields})"
        )
        print(f"[FALCON] P* = {self.k_selector.P_star}  (목표 충돌률)")
        for i, card in enumerate(self.field_cardinalities):
            mode = "compress" if self.budget.compressed_mask[i] else "raw-pass"
            print(
                f"[FALCON] Field {i:2d}: "
                f"card={card:>12,}, "
                f"k_hot={self.k_hot[i]:>2}, "
                f"k_cold={self.k_cold[i]:>2}, "
                f"M_hot={self.budget.M_hot[i]:>8,}, "
                f"M_cold={self.budget.M_cold[i]:>8,}, "
                f"M_f={self.budget.M_f[i]:>8,}, "
                f"pressure={self.budget.pressures[i]:.4f}, "
                f"mode={mode}"
            )
