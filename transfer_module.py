"""
CLS (Complementary Learning Systems) 기반 임베딩 이전 모듈.

LEAF의 Cold Start 문제:
    롱테일 → 숏헤드 전환 시 기존 임베딩 정보를 버리는 문제를 완화합니다.

CLS 이론 매핑:
    - 롱테일 테이블 (long_tail_hash) = 해마: 빠른 학습, 개별 패턴 보존
    - 숏헤드 테이블 (short_head_hash) = 신피질: 느린 학습, 안정적 패턴 유지
    - 전환 시 해마 → 신피질로 정보 이전 (통합 수면 단계에 해당)

양방향 동적 이전:
    - Forward  (저빈도 → 고빈도): 동적 alpha (학습 진행도에 따라 증가)
    - Backward (고빈도 → 저빈도): 동적 beta  (학습 진행도에 따라 감소)
      cold 행(한 번도 접근되지 않은 long-tail bucket row)에만 적용
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Set, Optional
from datasketches import frequent_items_error_type


class TransferModule:
    """
    빈도 전환(롱테일 ↔ 숏헤드) 시 임베딩 정보를 양방향으로 보존하는 모듈.

    Parameters
    ----------
    long_tail_hash : HashEmbedding
        롱테일 버킷 인덱스를 생성하는 HashEmbedding 객체.
    short_head_hash : HashEmbedding or None
        숏헤드 버킷 인덱스를 생성하는 HashEmbedding 객체.
        None이면 forward/backward 이전 모두 스킵합니다.
    emb_l : nn.ModuleList
        DLRM의 전체 임베딩 테이블 목록 (dlrm.emb_l).
    compressed_table_mask : np.ndarray
        각 테이블이 압축 대상인지 나타내는 bool 배열.
    ln_emb : np.ndarray
        전체 임베딩 카테고리 수 배열.
    selected_ln_emb_cum_offsets : torch.Tensor
        압축 테이블의 글로벌 인덱스 시작 오프셋 (CPU 텐서).
    online_frequency_checker : OnlineFrequencyChecker or None
        빈도 추적기. backward 이전(reverse_transfer)에 cold 행 탐색에 사용.
    alpha_min : float
        forward 이전의 최소 가중치 (배치 0).
    alpha_max : float
        forward 이전의 최대 가중치 (배치 total_batches).
    total_batches : int
        전체 학습 배치 수. progress 계산 기준.
    beta_max : float
        backward 이전의 최대 가중치 (배치 0, 학습 초반).
    beta_min : float
        backward 이전의 최소 가중치 (배치 total_batches, 학습 후반).
    reverse_freq : int
        backward 이전 실행 주기 (배치 단위).
    """

    def __init__(
        self,
        long_tail_hash,
        short_head_hash,
        emb_l: nn.ModuleList,
        compressed_table_mask: np.ndarray,
        ln_emb: np.ndarray,
        selected_ln_emb_cum_offsets: torch.Tensor,
        online_frequency_checker=None,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
        total_batches: int = 282891,
        beta_max: float = 0.2,
        beta_min: float = 0.05,
        reverse_freq: int = 1000,
    ):
        self.long_tail_hash = long_tail_hash
        self.short_head_hash = short_head_hash
        self.emb_l = emb_l
        self.online_frequency_checker = online_frequency_checker

        # forward alpha 스케줄
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha = alpha_min  # 현재 배치에서 사용할 alpha (update()에서 갱신)

        # backward beta 스케줄
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.reverse_freq = reverse_freq

        self.total_batches = total_batches

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

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def detect_transitions(self, current_short_head_set: Set[int]) -> Set[int]:
        """이번 배치에서 새롭게 숏헤드가 된 인덱스 집합 반환."""
        return current_short_head_set - self._prev_short_head_set

    def transfer_embeddings(
        self,
        newly_frequent: Set[int],
        device: torch.device,
    ) -> int:
        """
        [Forward] 롱테일 버킷 임베딩 → 숏헤드 버킷으로 가중 복사.

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

    def reverse_transfer(self, device: torch.device, current_batch: int) -> None:
        """
        [Backward] 숏헤드 평균 임베딩 → cold 롱테일 행 초기화.

        reverse_freq 배치마다 실행.

        Cold 행 정의
        -----------
        각 압축 테이블에서, 지금까지 한 번도 접근(hash)된 적 없는
        long-tail bucket row. FrequencyChecker 스케치에 기록된
        글로벌 인덱스를 warm으로 간주하고, 나머지를 cold로 처리.

        이전 공식
        ---------
            cold_row = (1 - beta) * cold_row + beta * sh_mean
        """
        if current_batch % self.reverse_freq != 0:
            return
        if self.short_head_hash is None or self.online_frequency_checker is None:
            return

        # 동적 beta: 학습 초반에 크고, 후반에 작아짐
        progress = current_batch / max(self.total_batches, 1)
        beta = self.beta_max - (self.beta_max - self.beta_min) * progress

        # 스케치에서 관찰된 글로벌 인덱스 수집 (NO_FALSE_POSITIVES → warm 인덱스)
        seen_items = self.online_frequency_checker.sketch.get_frequent_items(
            err_type=frequent_items_error_type.NO_FALSE_POSITIVES,
            threshold=0,
        )
        seen_global_indices = {item[0] for item in seen_items}

        # HashEmbedding의 row 범위 계산
        # long_tail: row [1, bins_lt - 1]  (short_head_offsets = 0)
        # short_head: row [bins_lt + 1, bins_lt + bins_sh - 1]
        lt_offset = int(self.long_tail_hash.short_head_offsets)   # 0
        lt_bins = self.long_tail_hash.bins
        sh_offset = int(self.short_head_hash.short_head_offsets)  # = bins_lt
        sh_bins = self.short_head_hash.bins

        all_lt_rows = set(range(1 + lt_offset, lt_bins + lt_offset))
        all_sh_rows = list(range(1 + sh_offset, sh_bins + sh_offset))

        total_cold = 0
        with torch.no_grad():
            for pos in range(self.num_compressed):
                k = self.compressed_table_indices[pos]
                emb_bag = self.emb_l[k]
                n_rows = emb_bag.weight.shape[0]

                start = int(self._cum_offsets[pos])
                end = start + int(self._table_sizes[pos])

                # 이 테이블 범위에 속하는 warm 글로벌 인덱스
                table_seen = [g for g in seen_global_indices if start <= g < end]

                # warm 글로벌 인덱스 → 접근된 lt_bucket row 집합
                if table_seen:
                    seen_tensor = torch.tensor(
                        table_seen, dtype=torch.long, device=device
                    )
                    lt_buckets = self.long_tail_hash.get_hash_embedding_tensors(
                        seen_tensor
                    )
                    # shape: [num_hashes, len(table_seen)] → flatten
                    warm_rows = set(lt_buckets.cpu().flatten().tolist())
                else:
                    warm_rows = set()

                # cold = 전체 lt rows - warm rows, 유효 범위(n_rows)로 클리핑
                cold_rows = sorted(
                    r for r in (all_lt_rows - warm_rows) if 0 <= r < n_rows
                )
                if not cold_rows:
                    continue

                # 유효한 숏헤드 행만 필터링
                valid_sh_rows = [r for r in all_sh_rows if 0 <= r < n_rows]
                if not valid_sh_rows:
                    continue

                sh_rows_t = torch.tensor(valid_sh_rows, dtype=torch.long, device=device)
                cold_rows_t = torch.tensor(cold_rows, dtype=torch.long, device=device)

                sh_mean = emb_bag.weight.data[sh_rows_t].mean(dim=0)
                emb_bag.weight.data[cold_rows_t] = (
                    (1.0 - beta) * emb_bag.weight.data[cold_rows_t]
                    + beta * sh_mean
                )
                total_cold += len(cold_rows)

        print(
            f"[CLS] 고→저 cold 행 초기화 완료 "
            f"(cold_count={total_cold}, beta={beta:.3f})"
        )

    def update(
        self,
        current_short_head_set: Set[int],
        device: torch.device,
        current_batch: int = 0,
    ) -> int:
        """
        매 배치마다 호출.

        1. 학습 진행도에 따라 alpha 동적 갱신
           alpha = alpha_min + (alpha_max - alpha_min) * progress
        2. 전환 감지 (detect_transitions)
        3. [Forward] 임베딩 이전 (transfer_embeddings)
        4. [Backward] cold 행 초기화 (reverse_transfer, reverse_freq 주기)
        5. 이전 집합 갱신

        Returns
        -------
        int
            이번 배치에서 forward 이전된 특성 수.
        """
        progress = current_batch / max(self.total_batches, 1)
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * progress

        newly_frequent = self.detect_transitions(current_short_head_set)
        n_transferred = self.transfer_embeddings(newly_frequent, device)

        # backward 이전 (reverse_freq 배치마다)
        self.reverse_transfer(device, current_batch)

        # 다음 배치 비교를 위해 현재 집합 저장
        self._prev_short_head_set = set(current_short_head_set)
        return n_transferred

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
