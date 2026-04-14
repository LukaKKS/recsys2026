"""
Co-occurrence statistics across fields (background thread, CPU).
Used by Similarity-aware Hashing to estimate feature / field similarity.
"""

from __future__ import annotations

import math
import queue
import threading
from collections import defaultdict
from typing import Any, DefaultDict, List, Tuple

# Per-field local feature ids must be < FIELD_ID_OFFSET (Avazu-safe).
FIELD_ID_OFFSET = 1_000_000_000


def _global_id(field_idx: int, feat_id: int) -> int:
    return int(field_idx) * FIELD_ID_OFFSET + int(feat_id)


class CooccurrenceTracker:
    """Decay-weighted counts and pairwise co-counts within each sample."""

    def __init__(
        self,
        num_fields: int,
        max_track: int = 10000,
        decay: float = 0.99,
        queue_maxsize: int = 64,
    ):
        self.num_fields = int(num_fields)
        self.max_track = int(max_track)
        self.decay = float(decay)
        self.n_batches = 0

        self.count: DefaultDict[int, float] = defaultdict(float)
        self.co_count: DefaultDict[Tuple[int, int], float] = defaultdict(float)

        self._q: queue.Queue[Any] = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._update_sync(item)
            finally:
                self._q.task_done()

    def _maybe_prune(self) -> None:
        if len(self.co_count) <= self.max_track * 200:
            return
        for k in list(self.co_count.keys()):
            self.co_count[k] *= 0.5
        for k in list(self.count.keys()):
            self.count[k] *= 0.5

    def _update_sync(self, batch_cat_features: List[Any]) -> None:
        """batch_cat_features: length num_fields, each (batch_size,) CPU tensor or array."""
        self.n_batches += 1
        if len(batch_cat_features) != self.num_fields:
            return

        batch_size = int(batch_cat_features[0].shape[0])
        d = self.decay

        for s in range(batch_size):
            sample_features: List[int] = []
            for f in range(self.num_fields):
                row = batch_cat_features[f]
                feat_id = int(row[s].item()) if hasattr(row[s], "item") else int(row[s])
                gid = _global_id(f, feat_id)
                sample_features.append(gid)
                self.count[gid] = self.count[gid] * d + 1.0

            L = len(sample_features)
            for i in range(L):
                for j in range(i + 1, L):
                    a, b = sample_features[i], sample_features[j]
                    if a > b:
                        a, b = b, a
                    self.co_count[(a, b)] = self.co_count[(a, b)] * d + 1.0

        self._maybe_prune()

    def update_async(self, batch_cat_features: List[Any]) -> None:
        try:
            cpu_batch = []
            for t in batch_cat_features:
                if hasattr(t, "detach"):
                    cpu_batch.append(t.detach().cpu().contiguous())
                else:
                    cpu_batch.append(t)
            self._q.put_nowait(cpu_batch)
        except queue.Full:
            pass

    def wait_pending(self) -> None:
        self._q.join()

    def close(self) -> None:
        self._stop.set()

    def get_similarity(
        self, feat_A: int, feat_B: int, field_A: int, field_B: int
    ) -> float:
        gA = _global_id(field_A, feat_A)
        gB = _global_id(field_B, feat_B)
        if gA > gB:
            gA, gB = gB, gA
        co = self.co_count.get((gA, gB), 0.0)
        cA = self.count.get(gA, 1.0)
        cB = self.count.get(gB, 1.0)
        return float(co / (cA + cB - co + 1e-10))

    def get_field_similarity(
        self,
        field_i: int,
        field_j: int,
        batch_cat_features: List[Any],
        max_samples: int = 32,
    ) -> float:
        if len(batch_cat_features) != self.num_fields:
            return 0.0
        batch_size = int(batch_cat_features[0].shape[0])
        n = min(batch_size, max_samples)
        if n <= 0:
            return 0.0
        total = 0.0
        for s in range(n):

            def _get(fi: int, sidx: int) -> int:
                row = batch_cat_features[fi]
                v = row[sidx]
                return int(v.item()) if hasattr(v, "item") else int(v)

            total += self.get_similarity(
                _get(field_i, s), _get(field_j, s), field_i, field_j
            )
        return total / n

    def get_harmful_collision_rate(
        self,
        field_idx: int,
        M: int,
        k: int,
        batch_cat_features: List[Any],
    ) -> float:
        if M <= 0 or k < 1:
            return 0.0
        try:
            p_col = 1.0 - math.exp(
                -1.0 / max(float(M) ** max(k - 1, 1), 1e-12)
            )
        except OverflowError:
            p_col = 1.0
        avg_dissim = 0.0
        cnt = 0
        for other in range(self.num_fields):
            if other == field_idx:
                continue
            sim = self.get_field_similarity(field_idx, other, batch_cat_features)
            avg_dissim += 1.0 - sim
            cnt += 1
        if cnt == 0:
            return float(p_col)
        return float(p_col * (avg_dissim / cnt))
