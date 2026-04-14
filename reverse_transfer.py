"""
Reverse Transfer for CLS-LEAF:
hot(short-head) -> cold(long-tail) knowledge transfer within the same (compressed) field.

Design constraints (LEAF):
- Keep existing CLS-LEAF behavior unchanged unless --use-reverse-transfer is enabled.
- No extra memory beyond current batch (operate on batch-appearing features only).
- Gradient-free: directly updates embedding weights under torch.no_grad().

Important (LEAF adaptive encoding):
- short_head_indices_set stores GLOBAL indices in the compressed pool
  (local feature id + selected_ln_emb_cum_offsets[field]).
- HashEmbedding.get_hash_embedding_tensors(global_ids) returns k hashed row indices.
  Here we use the first hash digit as a representative row for similarity/transfer
  (cheap and stable); this is sufficient for a lightweight reverse consolidation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ReverseTransferStats:
    transferred_pairs: int
    transferred_by_field: Sequence[int]
    beta: float


class ReverseTransferModule:
    def __init__(
        self,
        num_fields: int,
        field_cardinalities: Sequence[int],
        beta_min: float = 0.05,
        beta_max: float = 0.3,
        sim_threshold: float = 0.5,
        max_hot: int = 64,
        max_cold: int = 128,
    ):
        self.num_fields = int(num_fields)
        self.field_cardinalities = list(map(int, field_cardinalities))
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.sim_threshold = float(sim_threshold)
        self.max_hot = int(max_hot)
        self.max_cold = int(max_cold)
        self.total_batches = 0

    def get_beta(self, progress: float) -> float:
        p = float(max(0.0, min(1.0, progress)))
        return float(self.beta_min + (self.beta_max - self.beta_min) * p)

    @torch.no_grad()
    def transfer(
        self,
        batch_cat_features: torch.Tensor,
        selected_ln_emb_offsets: torch.Tensor,
        short_head_indices_set: Set[int],
        shared_emb_weight: torch.Tensor,
        long_tail_hash,
        short_head_hash,
        progress: float,
        device: torch.device,
    ) -> ReverseTransferStats:
        """
        Parameters
        ----------
        batch_cat_features:
            Tensor [num_compressed_fields, batch_size] of LOCAL feature ids.
        selected_ln_emb_offsets:
            Tensor [num_compressed_fields] global offsets for each compressed field.
        short_head_indices_set:
            Set of GLOBAL ids classified as short-head (updated by adaptive encoding).
        shared_emb_weight:
            EmbeddingBag.weight.data tensor for the SHARED compressed table.
        long_tail_hash / short_head_hash:
            HashEmbedding objects (same as used by adaptive encoding).
        """
        beta = self.get_beta(progress)
        num_fields = int(batch_cat_features.shape[0])
        transferred_by_field = [0 for _ in range(num_fields)]
        transferred_pairs = 0

        offsets = selected_ln_emb_offsets.to(device=device, dtype=torch.long)
        sh_set = short_head_indices_set  # local alias

        for f in range(num_fields):
            local_ids = batch_cat_features[f].to(device=device, dtype=torch.long)
            if local_ids.numel() == 0:
                continue
            global_ids = local_ids + offsets[f]

            # unique batch features for stability & speed
            uniq = torch.unique(global_ids)
            if uniq.numel() == 0:
                continue

            # split hot/cold by membership in short_head_indices_set
            # (short_head_indices_set is Python set -> CPU membership)
            uniq_cpu = uniq.detach().cpu().tolist()
            hot_globals = [g for g in uniq_cpu if g in sh_set]
            cold_globals = [g for g in uniq_cpu if g not in sh_set]
            if not hot_globals or not cold_globals:
                continue

            # cap to keep similarity computation cheap
            if len(hot_globals) > self.max_hot:
                hot_globals = hot_globals[: self.max_hot]
            if len(cold_globals) > self.max_cold:
                cold_globals = cold_globals[: self.max_cold]

            hot_g = torch.tensor(hot_globals, device=device, dtype=torch.long)
            cold_g = torch.tensor(cold_globals, device=device, dtype=torch.long)

            # representative hashed row per feature: first hash digit
            hot_rows = short_head_hash.get_hash_embedding_tensors(hot_g)[0]  # [n_hot]
            cold_rows = long_tail_hash.get_hash_embedding_tensors(cold_g)[0]  # [n_cold]

            hot_emb = shared_emb_weight[hot_rows]   # [n_hot, dim]
            cold_emb = shared_emb_weight[cold_rows]  # [n_cold, dim]

            hot_norm = F.normalize(hot_emb, dim=1)
            cold_norm = F.normalize(cold_emb, dim=1)
            sim = hot_norm @ cold_norm.t()  # [n_hot, n_cold]

            h_idx, c_idx = torch.where(sim > self.sim_threshold)
            if h_idx.numel() == 0:
                continue

            # apply reverse transfer per matched pair
            # cold_row <- (1-beta)*cold_row + beta*hot_row
            for hi, ci in zip(h_idx.tolist(), c_idx.tolist()):
                cr = int(cold_rows[ci].item())
                hr = int(hot_rows[hi].item())
                shared_emb_weight[cr] = (1.0 - beta) * shared_emb_weight[cr] + beta * shared_emb_weight[hr]
                transferred_pairs += 1
                transferred_by_field[f] += 1

        return ReverseTransferStats(
            transferred_pairs=transferred_pairs,
            transferred_by_field=transferred_by_field,
            beta=beta,
        )

