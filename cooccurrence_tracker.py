"""
Lightweight field-level statistics (background thread, CPU).

Goal:
- Keep adaptive encoding compression logic unchanged.
- Provide similarity estimates to adjust k only (Similarity-aware Hashing).

Implementation notes:
- Track per-field feature frequency with decay (no feature-pair table).
- Compute field-pair similarity via weighted Jaccard over per-field frequency maps.
  This reduces work to O(num_fields^2) per k-update (e.g. 22x22=484 pairs).
"""

from __future__ import annotations

import math
import queue
import threading
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

# Per-field local feature ids must be < FIELD_ID_OFFSET (Avazu-safe).
FIELD_ID_OFFSET = 1_000_000_000


def _global_id(field_idx: int, feat_id: int) -> int:
    return int(field_idx) * FIELD_ID_OFFSET + int(feat_id)


class CooccurrenceTracker:
    """Decay-weighted per-field feature frequency maps (field-level similarity only)."""

    def __init__(
        self,
        num_fields: int,
        max_track: int = 10000,
        decay: float = 0.99,
        queue_maxsize: int = 64,
        max_samples_per_batch: int = 32,
    ):
        self.num_fields = int(num_fields)
        self.max_track = int(max_track)
        self.decay = float(decay)
        self.max_samples_per_batch = int(max_samples_per_batch)
        self.n_batches = 0

        # Per-field feature counts:
        # counts[field_idx][feat_id] = decayed count
        self.counts: List[DefaultDict[int, float]] = [
            defaultdict(float) for _ in range(self.num_fields)
        ]

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

    def _maybe_prune_field(self, field_idx: int) -> None:
        d = self.counts[field_idx]
        if len(d) <= self.max_track:
            return
        # Keep top max_track by count (approximate: repeated pruning).
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[: self.max_track]
        self.counts[field_idx] = defaultdict(float, {k: float(v) for k, v in items})

    def _update_sync(self, batch_cat_features: List[Any]) -> None:
        """batch_cat_features: length num_fields, each (batch_size,) CPU tensor or array."""
        self.n_batches += 1
        if len(batch_cat_features) != self.num_fields:
            return

        batch_size = int(batch_cat_features[0].shape[0])
        n = min(batch_size, self.max_samples_per_batch)
        if n <= 0:
            return

        d = self.decay
        for f in range(self.num_fields):
            row = batch_cat_features[f]
            cd = self.counts[f]
            for s in range(n):
                v = row[s]
                feat_id = int(v.item()) if hasattr(v, "item") else int(v)
                cd[feat_id] = cd[feat_id] * d + 1.0
            self._maybe_prune_field(f)

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

    def get_field_similarity(
        self,
        field_i: int,
        field_j: int,
        batch_cat_features: List[Any] | None = None,
        max_samples: int = 32,  # kept for backward compatibility (ignored)
    ) -> float:
        del batch_cat_features, max_samples
        fi = int(field_i)
        fj = int(field_j)
        if fi < 0 or fj < 0 or fi >= self.num_fields or fj >= self.num_fields:
            return 0.0
        if fi == fj:
            return 1.0

        a: Dict[int, float] = self.counts[fi]
        b: Dict[int, float] = self.counts[fj]
        if not a or not b:
            return 0.0
        # Weighted Jaccard: sum(min)/sum(max)
        if len(a) > len(b):
            a, b = b, a
        inter = 0.0
        suma = 0.0
        for k, va in a.items():
            suma += va
            vb = b.get(k)
            if vb is not None:
                inter += va if va < vb else vb
        sumb = float(sum(b.values()))
        union = (suma + sumb - inter) + 1e-10
        return float(inter / union)

    def get_harmful_collision_rate(
        self,
        field_idx: int,
        M: int,
        k: int,
        batch_cat_features: List[Any] | None = None,
    ) -> float:
        del batch_cat_features
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
            sim = self.get_field_similarity(field_idx, other, None)
            avg_dissim += 1.0 - sim
            cnt += 1
        if cnt == 0:
            return float(p_col)
        return float(p_col * (avg_dissim / cnt))
