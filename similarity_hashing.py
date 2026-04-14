"""
Adjust per-field k from co-occurrence similarity. LEAF uses one HashEmbedding:
unified k = max_k over fields (applied to lt/sh via HashEmbedding.update_k).
"""

from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cooccurrence_tracker import CooccurrenceTracker


class SimilarityAwareHash:
    """Higher cross-field similarity -> lower k; lower similarity -> higher k."""

    def __init__(
        self,
        num_fields: int,
        k_base: int = 12,
        k_min: int = 4,
        k_max: int = 32,
        sim_threshold: float = 0.3,
    ):
        self.num_fields = int(num_fields)
        self.k_base = int(k_base)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.sim_threshold = float(sim_threshold)
        self.k_fields: List[int] = [self.k_base] * self.num_fields
        self._last_avg_sims: List[float] = [0.0] * self.num_fields

    def update_k(
        self,
        cooc_tracker: "CooccurrenceTracker",
        batch_cat_features: List[Any],
        field_cardinalities: List[int],
        M_budgets: Optional[int] = None,
    ) -> None:
        del field_cardinalities, M_budgets  # reserved for future HCR-aware k

        for f in range(self.num_fields):
            avg_sim = 0.0
            cnt = 0
            for g in range(self.num_fields):
                if g == f:
                    continue
                sim = cooc_tracker.get_field_similarity(f, g, batch_cat_features)
                avg_sim += sim
                cnt += 1
            if cnt > 0:
                avg_sim /= cnt
            self._last_avg_sims[f] = float(avg_sim)

            dissim = 1.0 - avg_sim
            new_k = max(
                self.k_min,
                min(self.k_max, round(self.k_base * dissim * 1.5)),
            )
            self.k_fields[f] = int(new_k)

    def get_k(self, field_idx: int) -> int:
        return self.k_fields[int(field_idx)]

    def get_unified_k(self) -> int:
        return max(self.k_fields) if self.k_fields else self.k_base

    def print_stats(self, hcr: Optional[float] = None) -> None:
        print("[SIM-HASH] per-field k / avg cross-field similarity:")
        for f, k in enumerate(self.k_fields):
            sim = self._last_avg_sims[f] if f < len(self._last_avg_sims) else 0.0
            if sim >= self.sim_threshold:
                tag = "similar to other fields (shared collisions OK)"
            else:
                tag = "dissimilar from other fields (avoid collisions)"
            print(f"  Field {f:2d}: k={k:2d}  (avg_sim={sim:.3f}, {tag})")
        if hcr is not None:
            print(
                f"[SIM-HASH] HCR(approx)={hcr:.4f} "
                f"(lower => less harmful collision mass vs fixed-k LEAF baseline)"
            )
