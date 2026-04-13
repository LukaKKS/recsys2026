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

    # ------------------------------------------------------------------
    def compute_entropy(self, field_idx: int) -> float:
        """SMED frequent items로 field의 Shannon 엔트로피 근사 계산.

        SMED는 상위 빈도 항목만 추적하므로 실제 엔트로피를 과소 추정하지만,
        field 간 상대적 비교에는 충분히 유효함.

        Returns
        -------
        H ≥ 0.0  (SMED에 데이터 없으면 0.0)
        """
        items = self.sketches[field_idx].get_frequent_items(
            err_type=frequent_items_error_type.NO_FALSE_POSITIVES,
            threshold=0,
        )
        if not items:
            return 0.0

        counts = [item[1] for item in items]
        total = sum(counts)
        if total == 0:
            return 0.0

        entropy = 0.0
        for c in counts:
            p = c / total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy


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
# 3. AdaptiveKSelector  (load-based, CR-adaptive)
# ---------------------------------------------------------------------------

class AdaptiveKSelector:
    """load = card / M_f 기반으로 k를 자동 계산.

    설계 원칙:
        CR이 클수록 M_f가 작아지고 load가 커짐
        → k를 자동으로 늘려 충돌 보상
        → 사람이 CR마다 k를 바꿀 필요 없음

    수식:
        load_f = card_f / M_f
        k_f    = max(k_min, ceil(log10(load_f) × k_scale))

    k_scale=5 기준 예시:
        load=1       → k=2   (raw-pass 수준)
        load=10      → k=6   (log10=1  × 5 = 5  → 6)
        load=100     → k=10  (log10=2  × 5 = 10)
        load=742     → k=15  (log10=2.87 × 5 ≈ 15)  CR=100 Field 9
        load=7421    → k=19  (log10=3.87 × 5 ≈ 19)  CR=1000 Field 9
        load=100000  → k=25  (log10=5  × 5 = 25)

    P*-역산 방식(v2.1) 대비 장점:
        P* 역산: CR 변화 시 k가 거의 안 바뀜 (공식 특성상 k_min에 걸림)
        load 기반: CR ↑ → load ↑ → k 자동 증가 → 균형 유지
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 32,
        k_scale: float = 5.0,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.k_scale = k_scale

    # ------------------------------------------------------------------
    def compute_k(self, n_features: int, M: int) -> int:
        """load 기반 k 계산.

        Returns
        -------
        k = 1              : M >= n_features (raw-pass 신호)
        k ∈ [k_min, k_max] : 압축 field
        """
        if M <= 0:
            return self.k_max
        if M >= n_features:
            return 1        # raw-pass 신호 (FALCONEncoder에서 사용 안 됨)
        if n_features <= 0:
            return self.k_min

        load = n_features / max(M, 1)
        if load <= 1.0:
            return self.k_min

        try:
            k = math.ceil(math.log10(load) * self.k_scale)
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
        3. AdaptiveKSelector (load 기반) — k_f = ceil(log10(load) × k_scale)
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
        k_scale: float = 5.0,
        K_base: int = 256,
        phase2_start: int = 10000,
        device: Optional[torch.device] = None,
    ):
        self.field_cardinalities = field_cardinalities
        self.n_fields = len(field_cardinalities)
        self.M_total = M_total
        self.arch_sparse_feature_size = arch_sparse_feature_size
        self.device = device or torch.device("cpu")

        # Phase 2 상태
        self.phase2_start = phase2_start
        self.phase2_done = False
        self.batch_count = 0

        # ── 3개 모듈 초기화 ──────────────────────────────────────────────
        self.field_smed = FieldWiseSMED(field_cardinalities, K_base)

        self.budget = FieldBudgetAllocator(M_total, field_cardinalities)

        self.k_selector = AdaptiveKSelector(k_min=2, k_max=32, k_scale=k_scale)

        print("[FALCON] Phase 1 시작 (load 기반 k)")

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

        self.batch_count += 1

        # SMED 비동기 업데이트 (GPU 학습과 병렬)
        self._pending_futures = self.field_smed.update_async(batch_cat_features)

        # Phase 2 체크: SMED warm-up 후 1회만 k 재조정
        if self.batch_count >= self.phase2_start and not self.phase2_done:
            # 이전 배치들의 SMED 업데이트 완료 대기 후 엔트로피 계산
            if self._pending_futures:
                futures_wait(self._pending_futures)
                self._pending_futures = []
            self._apply_phase2()

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
    def _apply_phase2(self) -> None:
        """Phase 2: SMED 엔트로피 기반 k 재조정 (1회만).

        1. 각 field의 Shannon 엔트로피를 SMED에서 추정
        2. 엔트로피 비례로 k 보정:
               k_new = k_old × (H_f / H_mean)
        3. HashEmbedding의 hash 파라미터만 재생성 (embedding.weight 보존)
           → cold start 없음

        엔트로피가 높은 field (feature가 고르게 분포):
            충돌 위험 더 큼 → k 증가
        엔트로피가 낮은 field (소수 feature가 대부분 담당):
            사실상 몇 개 feature만 자주 등장 → k 감소 가능
        """
        print(
            f"[FALCON] Phase 2 시작 "
            f"(배치 {self.batch_count}, 엔트로피 기반 k 재조정)"
        )

        # field별 엔트로피 계산 + 상세 출력
        entropies: List[float] = [
            self.field_smed.compute_entropy(i)
            for i in range(self.n_fields)
        ]

        n_zero = 0
        for i, h in enumerate(entropies):
            if not self.budget.compressed_mask[i]:
                continue
            if h == 0.0:
                n_zero += 1
                print(
                    f"[FALCON] Phase2 Field {i:2d}: "
                    f"H=0.000 ⚠️  SMED 데이터 부족 (k 유지)"
                )
            else:
                print(
                    f"[FALCON] Phase2 Field {i:2d}: "
                    f"H={h:.3f}, k_current={self.k_f[i]}"
                )

        # 압축 field 중 엔트로피 > 0인 것만 평균 계산
        valid_H = [h for i, h in enumerate(entropies)
                   if self.budget.compressed_mask[i] and h > 0]
        if not valid_H:
            print("[FALCON] Phase 2: 모든 압축 field SMED 데이터 부족 → k 유지")
            self.phase2_done = True
            return
        if n_zero > 0:
            print(
                f"[FALCON] Phase 2: {n_zero}개 field SMED 부족 "
                f"→ 해당 field k 유지, 나머지 조정 진행"
            )

        h_mean = sum(valid_H) / len(valid_H)
        print(f"[FALCON] Phase2 H_mean={h_mean:.3f} (압축 field 평균)")

        n_changed = 0
        for i in range(self.n_fields):
            if not self.budget.compressed_mask[i]:
                continue
            h_f = entropies[i]
            if h_f == 0:
                continue  # SMED 데이터 없는 field: k 유지

            # 엔트로피 비례 k 보정
            # ratio가 매우 작으면 k가 k_min까지 떨어져 충돌 폭발 가능
            # → Phase 1 load 기반 k는 하한선: 절대 현재값보다 줄이지 않음
            ratio = h_f / h_mean
            k_current = self.k_f[i]
            new_k = max(
                k_current,
                min(self.k_selector.k_max, round(k_current * ratio)),
            )

            if new_k != self.k_f[i]:
                # hash 파라미터만 재생성, embedding.weight 보존
                self.embeddings[i].update_k(new_k)
                print(
                    f"[FALCON] Phase2 Field {i:2d}: "
                    f"k {self.k_f[i]} → {new_k} "
                    f"(H={h_f:.3f}, ratio={ratio:.2f})"
                )
                self.k_f[i] = new_k
                n_changed += 1

        print(
            f"[FALCON] Phase 2 완료 "
            f"({n_changed}/{self.n_fields}개 field k 조정, "
            f"embedding.weight 보존)"
        )
        self.phase2_done = True

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
        print(
            f"[FALCON] k_scale={self.k_selector.k_scale}  "
            f"(k = ceil(log10(load) × k_scale), load = card / M_f)"
        )

        for i, card in enumerate(self.field_cardinalities):
            M = self.budget.M_f[i]
            k = self.k_f[i]
            mode = "compress" if self.budget.compressed_mask[i] else "raw-pass"
            load = card / max(M, 1)

            print(
                f"[FALCON] Field {i:2d}: "
                f"card={card:>12,}, "
                f"load={load:>8.1f}, "
                f"k={k:>2}, "
                f"M={M:>8,}, "
                f"pressure={self.budget.pressures[i]:.4f}, "
                f"mode={mode}"
            )
