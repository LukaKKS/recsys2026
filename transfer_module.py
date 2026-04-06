"""
CLS (Complementary Learning Systems) 기반 임베딩 이전 모듈.

LEAF의 Cold Start 문제:
    롱테일 → 숏헤드 전환 시 기존 임베딩 정보를 버리는 문제를 완화합니다.

CLS 이론 매핑:
    - 롱테일 테이블 (long_tail_hash) = 해마: 빠른 학습, 개별 패턴 보존
    - 숏헤드 테이블 (short_head_hash) = 신피질: 느린 학습, 안정적 패턴 유지
    - 전환 시 해마 → 신피질로 정보 이전 (통합 수면 단계에 해당)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Set, Optional


class TransferModule:
    """
    빈도 전환(롱테일 → 숏헤드) 시 임베딩 정보를 보존하는 모듈.

    Parameters
    ----------
    long_tail_hash : HashEmbedding
        롱테일 버킷 인덱스를 생성하는 HashEmbedding 객체.
    short_head_hash : HashEmbedding or None
        숏헤드 버킷 인덱스를 생성하는 HashEmbedding 객체.
        None이면 이전 동작 없이 집합 추적만 수행합니다.
    emb_l : nn.ModuleList
        DLRM의 전체 임베딩 테이블 목록 (dlrm.emb_l).
    compressed_table_mask : np.ndarray
        각 테이블이 압축 대상인지 나타내는 bool 배열.
    ln_emb : np.ndarray
        전체 임베딩 카테고리 수 배열.
    selected_ln_emb_cum_offsets : torch.Tensor
        압축 테이블의 글로벌 인덱스 시작 오프셋 (CPU 텐서).
    alpha_min : float
        학습 초반(배치 0)에 사용할 최소 alpha.
    alpha_max : float
        학습 후반(배치 total_batches)에 사용할 최대 alpha.
    total_batches : int
        전체 학습 배치 수. progress 계산 기준.
    """

    def __init__(
        self,
        long_tail_hash,
        short_head_hash,
        emb_l: nn.ModuleList,
        compressed_table_mask: np.ndarray,
        ln_emb: np.ndarray,
        selected_ln_emb_cum_offsets: torch.Tensor,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
        total_batches: int = 282891,
    ):
        self.long_tail_hash = long_tail_hash
        self.short_head_hash = short_head_hash
        self.emb_l = emb_l
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.total_batches = total_batches
        self.alpha = alpha_min  # 현재 배치에서 사용할 alpha (update()에서 갱신)

        # 압축 테이블의 emb_l 내 실제 인덱스 목록
        self.compressed_table_indices: list[int] = [
            k for k, is_comp in enumerate(compressed_table_mask) if is_comp
        ]
        self.num_compressed = len(self.compressed_table_indices)

        # 글로벌 인덱스 → 테이블 매핑에 필요한 오프셋/크기
        self._cum_offsets = selected_ln_emb_cum_offsets.cpu().tolist()
        self._table_sizes = ln_emb[compressed_table_mask].tolist()

        # 이전 배치의 숏헤드 집합 (초기값: 빈 집합)
        self._prev_short_head_set: Set[int] = set()

        # 직전 배치에서 새롭게 숏헤드가 된 인덱스 집합
        # adaptive_encoding에서 surge-k 해싱에 사용
        self.newly_frequent: Set[int] = set()

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def detect_transitions(self, current_short_head_set: Set[int]) -> Set[int]:
        """이번 배치에서 새롭게 숏헤드가 된 인덱스 집합 반환.

        Dynamic SMED decay 이후 apply_decay()가 short_head_indices_set에서
        항목을 제거하면, _prev_short_head_set에서도 해당 항목만 제거해야
        다음 배치에서 재진입 시 재전환으로 올바르게 감지된다.
        (dlrm_s_pytorch.py 학습 루프에서 decayed 항목을 직접 차감)
        """
        return current_short_head_set - self._prev_short_head_set

    def remove_from_prev_set(self, decayed: Set[int]) -> None:
        """decay된 항목만 _prev_short_head_set에서 제거.

        apply_decay()로 제거된 항목(decayed)을 prev_set에서만 빼주면,
        안정적인 항목은 prev_set에 그대로 유지되면서
        decay된 항목이 재진입할 때 정확히 재전환으로 감지된다.
        """
        self._prev_short_head_set -= decayed

    def transfer_embeddings(
        self,
        newly_frequent: Set[int],
        device: torch.device,
    ) -> int:
        """
        롱테일 버킷 임베딩 → 숏헤드 버킷으로 가중 복사.

        이전 공식:
            new_sh = alpha * lt_vec + (1 - alpha) * old_sh_vec

        Returns
        -------
        int
            실제로 이전된 특성(글로벌 인덱스) 수.
        """
        if not newly_frequent or self.short_head_hash is None:
            return 0

        transferred = 0
        with torch.no_grad():
            for g_idx in newly_frequent:
                table_pos, local_idx = self._global_to_table(g_idx)
                if table_pos is None:
                    continue

                k = self.compressed_table_indices[table_pos]
                emb_bag = self.emb_l[k]  # nn.EmbeddingBag
                n_rows = emb_bag.weight.shape[0]

                # device 일치 보장
                g_tensor = torch.tensor([g_idx], dtype=torch.long, device=device)

                # 각 HashEmbedding에서 버킷 인덱스 추출
                # shape: [num_hashes, 1]
                lt_buckets = self.long_tail_hash.get_hash_embedding_tensors(g_tensor)
                sh_buckets = self.short_head_hash.get_hash_embedding_tensors(g_tensor)

                # num_hashes 차원을 순회하며 이전
                n_hashes = lt_buckets.shape[0]
                for h in range(n_hashes):
                    lt_b = int(lt_buckets[h, 0].item())
                    sh_b = int(sh_buckets[h, 0].item())

                    if 0 <= lt_b < n_rows and 0 <= sh_b < n_rows:
                        lt_vec = emb_bag.weight.data[lt_b].clone()
                        sh_vec = emb_bag.weight.data[sh_b]
                        emb_bag.weight.data[sh_b] = (
                            self.alpha * lt_vec + (1.0 - self.alpha) * sh_vec
                        )

                transferred += 1

        return transferred

    def update(
        self,
        current_short_head_set: Set[int],
        device: torch.device,
        current_batch: int = 0,
    ) -> Set[int]:
        """
        매 배치마다 호출.

        1. 학습 진행도에 따라 alpha 동적 갱신
           alpha = alpha_min + (alpha_max - alpha_min) * progress
        2. 전환 감지 (detect_transitions)
        3. 임베딩 이전 (transfer_embeddings)
        4. 다음 배치를 위해 전환 집합 저장 (self.newly_frequent)
        5. 이전 집합 갱신

        Returns
        -------
        Set[int]
            이번 배치에서 새롭게 숏헤드가 된 글로벌 인덱스 집합.
            다음 배치의 surge-k 해싱에 활용하기 위해 self.newly_frequent에도 저장.
        """
        progress = current_batch / max(self.total_batches, 1)
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * progress

        newly_frequent = self.detect_transitions(current_short_head_set)

        # decay 직후 구간(배치 9990~10200)에서 집합 크기 추적
        if 9990 <= current_batch <= 10200:
            print(f"[CLS-DEBUG] batch={current_batch} "
                  f"current_set={len(current_short_head_set)} "
                  f"prev_set={len(self._prev_short_head_set)} "
                  f"newly_frequent={len(newly_frequent)}")

        self.transfer_embeddings(newly_frequent, device)

        # 다음 배치의 surge-k 해싱을 위해 전환 집합 보존
        self.newly_frequent = newly_frequent

        # 다음 배치 비교를 위해 현재 집합 저장
        self._prev_short_head_set = set(current_short_head_set)
        return newly_frequent

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _global_to_table(self, global_idx: int):
        """
        글로벌 인덱스 → (압축 테이블 위치, 로컬 인덱스) 변환.

        압축 테이블은 `selected_ln_emb_cum_offsets`로 순서가 매겨져 있고,
        각 테이블의 인덱스 범위는 [cum_offset[i], cum_offset[i] + size[i]) 입니다.
        """
        for pos in range(self.num_compressed):
            start = int(self._cum_offsets[pos])
            end = start + int(self._table_sizes[pos])
            if start <= global_idx < end:
                return pos, global_idx - start
        return None, None
