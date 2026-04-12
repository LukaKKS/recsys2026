"""
FALCON: Field-Aware Lightweight COmpression with pressure-derived policy

각 field의 통계적 특성(카디널리티, skewness, mass)을 분석해
field마다 최적의 hash bucket 수(M_f)와 hash 함수 수(k_f)를 자동으로 결정한다.

Components:
    FieldPressureEstimator  : field별 compression pressure 계산
    FieldWiseSMED           : field별 독립 빈도 스케치
    AdaptiveKSelector       : field별 최적 k (hash 수) 결정
    BudgetAllocator         : pressure 기반 메모리 배분
    FALCONEncoder           : 전체 FALCON 파이프라인
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from typing import List, Dict, Optional, Tuple

from datasketches import frequent_items_sketch, frequent_items_error_type
from hash_embedding import HashEmbedding


# ---------------------------------------------------------------------------
# 1. FieldPressureEstimator
# ---------------------------------------------------------------------------

class FieldPressureEstimator:
    """각 field의 compression pressure 계산.

    Pressure_f = α·log(|X_f|) + β·skew_f + γ·(1 - mass_f)

    초기 버전은 카디널리티 기반 정규화:
        P_f = log(|X_f|) / Σ log(|X_i|)
    학습이 진행되면서 skewness와 mass를 반영해 pressure를 조정한다.
    """

    ALPHA = 0.5   # cardinality 가중치
    BETA  = 0.3   # skewness 가중치
    GAMMA = 0.2   # tail-mass 가중치
    UPDATE_INTERVAL = 1000  # 몇 배치마다 skew/mass 재계산

    def __init__(self, field_cardinalities: List[int]):
        self.cardinalities = field_cardinalities
        self.n_fields = len(field_cardinalities)

        # 기본 pressure: 정규화된 log(cardinality)
        log_cards = [math.log(max(c, 2)) for c in field_cardinalities]
        total_log = sum(log_cards)
        self._base_pressures: List[float] = [lc / total_log for lc in log_cards]

        # 빈도 카운터 (skewness/mass 추정용)
        self._counters: List[Counter] = [Counter() for _ in range(self.n_fields)]
        self._update_counts: List[int] = [0] * self.n_fields

        # 조정된 pressure (skew/mass 반영 후)
        self._adjusted_pressures: List[float] = list(self._base_pressures)

    # ------------------------------------------------------------------
    def compute_pressure(self, field_idx: int) -> float:
        return self._adjusted_pressures[field_idx]

    def get_all_pressures(self) -> List[float]:
        return list(self._adjusted_pressures)

    # ------------------------------------------------------------------
    def update_statistics(self, field_idx: int, field_values) -> None:
        """배치에서 field 값들의 빈도 분포 업데이트."""
        for v in field_values:
            self._counters[field_idx][int(v)] += 1
        self._update_counts[field_idx] += 1

        if self._update_counts[field_idx] % self.UPDATE_INTERVAL == 0:
            self._recompute_pressure(field_idx)

    # ------------------------------------------------------------------
    def _recompute_pressure(self, field_idx: int) -> None:
        """skewness와 mass를 반영해 pressure 재계산 후 전체 정규화."""
        counter = self._counters[field_idx]
        if len(counter) < 2:
            return

        counts = np.array(list(counter.values()), dtype=float)
        mean, std = counts.mean(), counts.std()
        if std < 1e-8:
            return

        # skewness (간단한 근사)
        skewness = float(np.mean(((counts - mean) / std) ** 3))

        # mass: 상위 10% 특성이 차지하는 빈도 비율
        sorted_counts = np.sort(counts)[::-1]
        top_n = max(1, len(sorted_counts) // 10)
        mass = float(sorted_counts[:top_n].sum() / counts.sum())

        log_card = math.log(max(self.cardinalities[field_idx], 2))
        raw = (
            self.ALPHA * log_card
            + self.BETA  * abs(skewness)
            + self.GAMMA * (1.0 - mass)
        )

        # 해당 field만 업데이트하고 전체 재정규화
        raw_all = []
        for i in range(self.n_fields):
            if i == field_idx:
                raw_all.append(raw)
            else:
                lc = math.log(max(self.cardinalities[i], 2))
                raw_all.append(self.ALPHA * lc)   # 나머지는 기본값 사용
        total = sum(raw_all)
        if total > 0:
            self._adjusted_pressures = [r / total for r in raw_all]


# ---------------------------------------------------------------------------
# 2. FieldWiseSMED
# ---------------------------------------------------------------------------

class FieldWiseSMED:
    """field마다 독립적인 SMED (Sketch-based Memory-Efficient Dictionary) sketch.

    K_f = K_base × max(1, int(log2(|X_f|)))
    """

    def __init__(self, field_cardinalities: List[int], K_base: int = 256):
        self.n_fields = len(field_cardinalities)
        self.cardinalities = field_cardinalities
        self.K_base = K_base

        # 각 field별 K 계산
        self.K_per_field: List[int] = []
        for card in field_cardinalities:
            log2_card = max(1, int(math.log2(max(card, 2))))
            self.K_per_field.append(K_base * log2_card)

        # 각 field별 독립 sketch
        # lg_max_k = max(4, int(log2(K_f))) → sketch 용량
        self.sketches: List[frequent_items_sketch] = []
        for K_f in self.K_per_field:
            lg_max_k = max(4, int(math.log2(max(K_f, 16))))
            self.sketches.append(frequent_items_sketch(lg_max_k))

        # field별 short-head 집합 (고빈도 특성)
        self.short_head_sets: List[set] = [set() for _ in range(self.n_fields)]

    # ------------------------------------------------------------------
    def update(self, field_idx: int, field_values) -> None:
        """해당 field의 sketch 업데이트."""
        for v in field_values:
            self.sketches[field_idx].update(int(v))

    # ------------------------------------------------------------------
    def get_frequent(
        self, field_idx: int, percentile: float = 90.0
    ) -> set:
        """해당 field의 고빈도 feature 반환."""
        sketch = self.sketches[field_idx]
        items = sketch.get_frequent_items(
            err_type=frequent_items_error_type.NO_FALSE_POSITIVES,
            threshold=0,
        )
        if not items:
            return set()
        counts = [item[1] for item in items]
        threshold = float(np.percentile(counts, percentile))
        return {item[0] for item in items if item[1] >= threshold}

    # ------------------------------------------------------------------
    def get_frequency_estimation_error(self, field_idx: int) -> Dict:
        """단일 sketch vs field-wise sketch 추정 오차 계산 (실험용)."""
        return {
            "field_idx": field_idx,
            "active_items": self.sketches[field_idx].get_num_active_items(),
            "K_f": self.K_per_field[field_idx],
        }


# ---------------------------------------------------------------------------
# 3. AdaptiveKSelector
# ---------------------------------------------------------------------------

class AdaptiveKSelector:
    """field별 최적 k (hash 함수 수) 자동 결정.

    k_f = argmin_k [P_col(f, k) + λ·Cost(k)]

        P_col(f, k) = max(0, log(card / M_f)) / k   ← 로드 팩터 기반 수식
            - card < M_f (여유): load < 1 → P_col = 0 → k_min 선택 (비용 절감)
            - card > M_f (과부하): P_col = log(load)/k → k가 클수록 감소
            - 해석: 버킷당 평균 충돌 수(load)의 로그를 k로 분산
            - 해석적 최적: k* ≈ sqrt(log(load) · k_max / λ)

        Cost(k) = k / k_max   (정규화된 연산 비용)

    이전 수식 P_col = 1 - exp(-card/M^k) 문제:
        M^k가 지수적으로 폭발 → P_col ≈ 0 for any k ≥ 2 →
        optimizer가 항상 k_min=2 선택 (의미 없는 grid search)
    """

    def __init__(self, lambda_cost: float = 0.1, k_min: int = 2, k_max: int = 32):
        self.lambda_cost = lambda_cost
        self.k_min = k_min
        self.k_max = k_max

    # ------------------------------------------------------------------
    def compute_optimal_k(
        self, cardinality: int, memory_budget: int
    ) -> int:
        """최적 k 탐색 (grid search over [k_min, k_max]).

        P_col(k) = max(0, log(card / M)) / k
            - 과부하(card > M): 로드 팩터 로그를 k로 분산
            - 여유(card ≤ M): P_col = 0, cost만 남아 k_min 선택
        """
        M = max(memory_budget, 1)
        card = max(cardinality, 1)
        load = card / M

        # 여유 상태: 어떤 k를 써도 충돌 없음 → 최소 비용(k_min) 선택
        if load <= 1.0:
            return self.k_min

        log_load = math.log(load)   # load > 1 이므로 양수

        best_k = self.k_min
        best_loss = float("inf")

        for k in range(self.k_min, self.k_max + 1):
            p_col = log_load / k        # k가 클수록 작아짐
            cost = k / self.k_max
            loss = p_col + self.lambda_cost * cost

            if loss < best_loss:
                best_loss = loss
                best_k = k

        return best_k


# ---------------------------------------------------------------------------
# 4. BudgetAllocator
# ---------------------------------------------------------------------------

class BudgetAllocator:
    """field pressure 기반 메모리 배분.

    M_f = M_total × pressure_f / Σ pressure_i
    최소 min_rows 보장.
    """

    def __init__(
        self,
        M_total: int,
        field_cardinalities: List[int],
        min_rows: int = 100,
    ):
        self.M_total = M_total
        self.field_cardinalities = field_cardinalities
        self.n_fields = len(field_cardinalities)
        self.min_rows = min_rows

    # ------------------------------------------------------------------
    def allocate(self, pressures: List[float]) -> List[int]:
        """pressure 비례 배분 + min_rows 보장."""
        total_pressure = sum(pressures)

        if total_pressure < 1e-10:
            base = max(self.min_rows, self.M_total // self.n_fields)
            return [base] * self.n_fields

        # 비례 배분
        allocations = [
            max(self.min_rows, int(self.M_total * p / total_pressure))
            for p in pressures
        ]

        # 합계가 M_total 초과 시 초과분 삭감 (min_rows 보호)
        total_alloc = sum(allocations)
        if total_alloc > self.M_total:
            excess = total_alloc - self.M_total
            reducible = [a - self.min_rows for a in allocations]
            total_reducible = sum(reducible)
            if total_reducible > 0:
                allocations = [
                    a - int(excess * r / total_reducible)
                    for a, r in zip(allocations, reducible)
                ]

        return allocations


# ---------------------------------------------------------------------------
# 5. FALCONEncoder
# ---------------------------------------------------------------------------

class FALCONEncoder:
    """전체 FALCON 파이프라인.

    field별로 독립적인 HashEmbedding을 생성하고,
    pressure 기반으로 메모리를 배분하며,
    adaptive k 선택으로 hash 충돌을 최소화한다.
    """

    def __init__(
        self,
        field_cardinalities: List[int],
        M_total: int,
        arch_sparse_feature_size: int = 16,
        lambda_cost: float = 0.1,
        K_base: int = 256,
        device: Optional[torch.device] = None,
    ):
        self.field_cardinalities = field_cardinalities
        self.n_fields = len(field_cardinalities)
        self.M_total = M_total
        self.arch_sparse_feature_size = arch_sparse_feature_size
        self.device = device or torch.device("cpu")

        # 구성 요소 초기화
        self.pressure_estimator = FieldPressureEstimator(field_cardinalities)
        self.field_smed = FieldWiseSMED(field_cardinalities, K_base)
        self.k_selector = AdaptiveKSelector(lambda_cost)
        self.budget_allocator = BudgetAllocator(M_total, field_cardinalities)

        # 초기 pressure로 budget 계산
        pressures = self.pressure_estimator.get_all_pressures()
        self.field_budgets: List[int] = self.budget_allocator.allocate(pressures)

        # 각 field별 최적 k 계산
        self.field_k: List[int] = [
            self.k_selector.compute_optimal_k(card, budget)
            for card, budget in zip(field_cardinalities, self.field_budgets)
        ]

        # 각 field별 HashEmbedding 생성
        # FALCON은 field별 독립 EmbeddingBag(크기 M_f)을 사용하므로
        # HashEmbedding의 출력 인덱스는 반드시 [0, M_f-1] 범위여야 한다.
        # → offsets=0 (출력 시프트 없음)
        #
        # LEAF의 shared-table 방식과 달리, FALCON은 field별 테이블을 분리하므로
        # offsets로 shared table 내 위치를 지정할 필요가 없다.
        # HashEmbedding의 hash 함수 파라미터(multiplers, adders)가
        # field별로 달라지므로 같은 raw 값도 다른 bucket으로 해싱된다.
        self.field_hash_embeddings: List[HashEmbedding] = []
        for i, (card, budget, k) in enumerate(
            zip(field_cardinalities, self.field_budgets, self.field_k)
        ):
            he = HashEmbedding(
                card,                   # field의 카디널리티 (input space = 해당 field만)
                arch_sparse_feature_size,
                num_buckets=budget,     # field별 독립 bucket 수 M_f
                num_hashes=k,
                append_weight=False,
                offsets=0,              # 출력 인덱스 [0, M_f-1] 그대로 유지
                device=self.device,
            )
            self.field_hash_embeddings.append(he)

        self.batch_count: int = 0

    # ------------------------------------------------------------------
    def encode_field(
        self, field_idx: int, local_indices: torch.Tensor
    ) -> torch.Tensor:
        """단일 field의 local 인덱스를 hash bucket 인덱스로 변환.

        Parameters
        ----------
        local_indices : (n,) 텐서 — field-local 0-indexed 값

        Returns
        -------
        (k_f, n) 텐서 — hash bucket 인덱스 (0 ~ M_f-1)
        """
        he = self.field_hash_embeddings[field_idx]
        return he.get_hash_embedding_tensors(local_indices.to(self.device))

    # ------------------------------------------------------------------
    def encode(
        self,
        batch_cat_features: List[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """전체 배치 인코딩.

        Parameters
        ----------
        batch_cat_features : list[Tensor(batch_size,)]
            field별 raw(local) 인덱스 텐서 리스트.

        Returns
        -------
        encoded_indices : list[Tensor(batch_size * k_f,)]
            field별 flatten된 hash bucket 인덱스.
        encoded_offsets : list[Tensor(batch_size,)]
            field별 EmbeddingBag offset (0, k_f, 2k_f, ...).
        """
        dev = device or self.device
        encoded_indices: List[torch.Tensor] = []
        encoded_offsets: List[torch.Tensor] = []

        for field_idx, raw_idx in enumerate(batch_cat_features):
            raw_np = raw_idx.cpu().numpy()

            # SMED 및 pressure 업데이트
            self.pressure_estimator.update_statistics(field_idx, raw_np)
            self.field_smed.update(field_idx, raw_np)

            # [k_f, batch_size] → flatten
            hashed = self.encode_field(field_idx, raw_idx)  # [k_f, batch_size]
            k_f, batch_size = hashed.shape

            # 인덱스 범위 검증 (첫 10배치에서만)
            if self.batch_count < 10:
                M_f = self.field_budgets[field_idx]
                h_min, h_max = int(hashed.min()), int(hashed.max())
                if h_max >= M_f or h_min < 0:
                    print(
                        f"[FALCON WARNING] Field {field_idx}: "
                        f"hash index out of range! "
                        f"min={h_min}, max={h_max}, M_f={M_f}",
                        flush=True,
                    )

            # EmbeddingBag 형식: 각 샘플의 k_f개 hash를 하나의 bag으로
            # flat_indices: [batch_size * k_f] — 각 샘플의 k_f개 bucket 인덱스
            # offsets:      [batch_size]       — 각 샘플의 시작 위치 (0, k_f, 2k_f, ...)
            flat_indices = hashed.T.reshape(-1).to(dev)
            offsets = torch.arange(
                0, batch_size * k_f, k_f, dtype=torch.long, device=dev
            )

            encoded_indices.append(flat_indices)
            encoded_offsets.append(offsets)

        self.batch_count += 1
        return encoded_indices, encoded_offsets

    # ------------------------------------------------------------------
    def get_field_ln_emb(self) -> List[int]:
        """DLRM 모델 초기화용: field별 embedding table 크기 반환."""
        return list(self.field_budgets)

    # ------------------------------------------------------------------
    def print_field_stats(self) -> None:
        """field별 통계 출력."""
        pressures = self.pressure_estimator.get_all_pressures()
        print(
            f"[FALCON] 전체 budget 배분 완료 "
            f"(M_total={self.M_total}, fields={self.n_fields})"
        )
        for i in range(self.n_fields):
            # 로드 팩터 기반 P_col 수식 (AdaptiveKSelector와 동일)
            M_i = max(self.field_budgets[i], 1)
            k_i = self.field_k[i]
            card_i = max(self.field_cardinalities[i], 1)
            load_i = card_i / M_i
            p_col = max(0.0, math.log(load_i)) / k_i if load_i > 1.0 else 0.0
            print(
                f"[FALCON] Field {i:2d}: "
                f"card={self.field_cardinalities[i]:>12,}, "
                f"K={self.field_smed.K_per_field[i]:>6,}, "
                f"k={self.field_k[i]:>2}, "
                f"M={self.field_budgets[i]:>8,}, "
                f"pressure={pressures[i]:.4f}, "
                f"P_col≈{p_col:.4f}"
            )
