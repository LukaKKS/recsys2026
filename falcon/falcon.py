"""
FALCON: Field-Aware Lightweight COmpression with pressure-derived policy

각 field의 카디널리티를 분석해 field마다 최적의 hash bucket 수(M_f)와
hash 함수 수(k_f)를 학습 전 1회 계산하고 학습 중에는 추가 연산 없이
field별 adaptive k로 직접 해싱한다.

Zero training overhead:
    pressure / budget / k → 초기화 시 카디널리티 기반 1회 계산, 이후 고정.
    학습 중에는 field별 HashEmbedding으로 직접 해싱만 수행.

Components:
    FieldPressureEstimator  : field별 compression pressure 계산 (초기화 시 1회)
    AdaptiveKSelector       : field별 최적 k (hash 수) 결정 (초기화 시 1회)
    BudgetAllocator         : pressure 기반 메모리 배분 (초기화 시 1회)
    FALCONEncoder           : 전체 FALCON 파이프라인
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from hash_embedding import HashEmbedding


# ---------------------------------------------------------------------------
# 1. FieldPressureEstimator
# ---------------------------------------------------------------------------

class FieldPressureEstimator:
    """각 field의 compression pressure 계산.

    학습 전 카디널리티만으로 pressure를 계산하고 이후 고정.
    학습 중 추가 연산 없음 (zero training overhead).

        pressure_f = log(|X_f|) / Σ log(|X_i|)

    논문 메시지: "compression policy는 학습 전 field 통계로부터
    자동 유도되며 학습 중 추가 연산이 없다."
    """

    def __init__(self, field_cardinalities: List[int]):
        self.cardinalities = field_cardinalities
        self.n_fields = len(field_cardinalities)

        # 초기화 시 1회만 계산, 이후 고정
        log_cards = [math.log(max(c, 2)) for c in field_cardinalities]
        total_log = sum(log_cards)
        self._pressures: List[float] = [lc / total_log for lc in log_cards]

    # ------------------------------------------------------------------
    def compute_pressure(self, field_idx: int) -> float:
        return self._pressures[field_idx]

    def get_all_pressures(self) -> List[float]:
        return list(self._pressures)


# ---------------------------------------------------------------------------
# 2. AdaptiveKSelector
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
        device: Optional[torch.device] = None,
    ):
        self.field_cardinalities = field_cardinalities
        self.n_fields = len(field_cardinalities)
        self.M_total = M_total
        self.arch_sparse_feature_size = arch_sparse_feature_size
        self.device = device or torch.device("cpu")

        # 구성 요소 초기화 (초기화 시 1회, 학습 중 불변)
        self.pressure_estimator = FieldPressureEstimator(field_cardinalities)
        self.k_selector = AdaptiveKSelector(lambda_cost)
        self.budget_allocator = BudgetAllocator(M_total, field_cardinalities)

        # 초기 pressure로 budget 계산
        pressures = self.pressure_estimator.get_all_pressures()
        raw_budgets = self.budget_allocator.allocate(pressures)

        # 여유 field (card ≤ 배분된 budget)는 자연 크기(card)로 고정
        # → 압축 불필요, EmbeddingBag 낭비 없음
        self.field_budgets: List[int] = [
            min(card, budget)
            for card, budget in zip(field_cardinalities, raw_budgets)
        ]

        # 압축 필요 여부: card > M_f 인 field만 HashEmbedding 사용
        # 여유 field: raw indices 직접 통과 (LEAF baseline과 동일 속도)
        self.compressed_mask: List[bool] = [
            card > budget
            for card, budget in zip(field_cardinalities, self.field_budgets)
        ]

        # 각 field별 최적 k 계산 (압축 field만 의미 있음)
        self.field_k: List[int] = [
            self.k_selector.compute_optimal_k(card, budget) if compressed else 1
            for card, budget, compressed in zip(
                field_cardinalities, self.field_budgets, self.compressed_mask
            )
        ]

        n_compressed = sum(self.compressed_mask)
        print(
            f"[FALCON] 압축 대상 field: {n_compressed}/{self.n_fields}개 "
            f"(나머지 {self.n_fields - n_compressed}개는 raw indices 통과)"
        )

        # 압축 field에 대해서만 HashEmbedding 생성
        # → 여유 field는 encode()에서 raw indices를 직접 반환
        self.field_hash_embeddings: Dict[int, HashEmbedding] = {}
        for i, (card, budget, k, compressed) in enumerate(
            zip(field_cardinalities, self.field_budgets, self.field_k, self.compressed_mask)
        ):
            if not compressed:
                continue
            he = HashEmbedding(
                card,
                arch_sparse_feature_size,
                num_buckets=budget,
                num_hashes=k,
                append_weight=False,
                offsets=0,
                device=self.device,
            )
            self.field_hash_embeddings[i] = he

    # ------------------------------------------------------------------
    def encode(
        self,
        batch_cat_features: List[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """전체 배치 인코딩.

        압축 field (card > M_f): adaptive k-hash → flat_indices + k-step offsets
        여유 field (card ≤ M_f): raw indices 직접 통과 → EmbeddingBag 1-index-per-sample

        Parameters
        ----------
        batch_cat_features : list[Tensor(batch_size,)]
            field별 raw(local, 0-indexed) 인덱스 텐서 리스트.

        Returns
        -------
        encoded_indices : list[Tensor]
        encoded_offsets : list[Tensor]
        """
        dev = device or self.device
        encoded_indices: List[torch.Tensor] = []
        encoded_offsets: List[torch.Tensor] = []

        # 여유 field 공용 offsets (1-index-per-sample): 배치마다 1회 생성
        batch_size = batch_cat_features[0].shape[0]
        simple_offsets = torch.arange(batch_size, dtype=torch.long, device=dev)

        for field_idx, raw_idx in enumerate(batch_cat_features):
            if not self.compressed_mask[field_idx]:
                # 여유 field: raw indices 그대로 통과 (HashEmbedding 스킵)
                encoded_indices.append(raw_idx.to(dev))
                encoded_offsets.append(simple_offsets)
            else:
                # 압축 field: adaptive k 해싱
                he = self.field_hash_embeddings[field_idx]
                hashed = he.get_hash_embedding_tensors(raw_idx.to(self.device))
                k_f = hashed.shape[0]
                flat_indices = hashed.T.reshape(-1).to(dev)
                offsets = torch.arange(
                    0, batch_size * k_f, k_f, dtype=torch.long, device=dev
                )
                encoded_indices.append(flat_indices)
                encoded_offsets.append(offsets)

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
            M_i = max(self.field_budgets[i], 1)
            k_i = self.field_k[i]
            card_i = max(self.field_cardinalities[i], 1)
            load_i = card_i / M_i
            p_col = max(0.0, math.log(load_i)) / k_i if load_i > 1.0 else 0.0
            mode = "compress" if self.compressed_mask[i] else "raw-pass"
            print(
                f"[FALCON] Field {i:2d}: "
                f"card={self.field_cardinalities[i]:>12,}, "
                f"k={k_i:>2}, "
                f"M={self.field_budgets[i]:>8,}, "
                f"pressure={pressures[i]:.4f}, "
                f"P_col≈{p_col:.4f}, "
                f"mode={mode}"
            )
