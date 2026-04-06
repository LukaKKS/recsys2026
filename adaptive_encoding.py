import heapq 
import torch
import numpy as np
import pickle
import random
from collections import defaultdict
from typing import Dict, Optional
import time
import os
import ctypes
import logging
import math
from datasketches import kll_floats_sketch
from datasketches import kll_ints_sketch
from functools import partial
from itertools import permutations
from itertools import combinations
from hash_embedding import HashEmbedding
from collections import Counter
from datasketches import frequent_items_sketch
from datasketches import frequent_items_error_type


class OnlineFrequencyChecker:
    def __init__(self, lg_max_k=10):
        self.sketch = frequent_items_sketch(lg_max_k)
        self.lg_max_k = lg_max_k
        # {idx: remaining_batches_on_blacklist}
        # datasketches는 카운터 직접 수정 불가 → 블랙리스트로 재진입을 제어
        self._decay_blacklist: Dict[int, int] = {}

    def add_elements(self, elements):
        for item in elements:
            self.sketch.update(item.item())

    def get_frequency_percentile(self, percentile: float, short_head_indices_set=set()) -> int:
        # 블랙리스트 카운터 감소 및 만료 항목 제거
        if self._decay_blacklist:
            expired = [idx for idx, cnt in self._decay_blacklist.items() if cnt <= 0]
            for idx in expired:
                del self._decay_blacklist[idx]
            if expired:
                print(f"[CLS-DEBUG] 블랙리스트 만료: {len(expired)}개 → short_head 재진입 허용 "
                      f"(잔여 블랙리스트: {len(self._decay_blacklist)}개)")
            for idx in list(self._decay_blacklist.keys()):
                self._decay_blacklist[idx] -= 1

        frequent_items = self.sketch.get_frequent_items(
            err_type=frequent_items_error_type.NO_FALSE_POSITIVES,
            threshold=0
        )

        # 블랙리스트 항목 제외 (스케치 카운터가 높아도 일정 기간 재진입 차단)
        if self._decay_blacklist:
            before_count = len(frequent_items)
            frequent_items = [item for item in frequent_items
                              if item[0] not in self._decay_blacklist]
            after_count = len(frequent_items)
            print(f"[CLS-DEBUG] 필터링 전: {before_count}개 / "
                  f"블랙리스트 필터링 후: {after_count}개 "
                  f"(차단 중: {len(self._decay_blacklist)}개)")

        if not frequent_items:
            return 0, []

        top_index = max(0, int(len(frequent_items) * percentile) - 1)
        indices = [e[0] for e in frequent_items]
        short_head_indices_set.update(indices)
        top_percentile_frequency = frequent_items[top_index][1]
        short_head_indices = [e[0] for e in frequent_items[:top_index + 1]]
        return top_percentile_frequency, short_head_indices

    def apply_decay(
        self,
        short_head_indices_set: set,
        decay_rate: float,
        current_batch: int,
        decay_freq: int,
        min_size: int = 100,
        grace_period: int = 100,
    ) -> set:
        """
        Dynamic SMED: 주기적으로 short_head_indices_set에서
        저빈도 항목을 제거하여 재전환(re-transition)을 유도한다.

        동작 원리
        ---------
        1. 스케치에서 현재 빈도 정보를 읽는다.
        2. short_head_indices_set을 빈도 오름차순으로 정렬한다.
        3. 하위 (1 - decay_rate) 비율 항목을 집합에서 제거한다.
           (min_size 이하로는 제거하지 않는다.)
        4. 제거된 항목을 블랙리스트에 등록 (grace_period 배치 동안).
           → get_frequency_percentile에서 스케치 카운터가 높아도 재진입 차단
           → grace_period 경과 후 자연스럽게 재진입 → 재전환 감지!

        Parameters
        ----------
        short_head_indices_set : set
            현재 고빈도 집합 (in-place 수정).
        decay_rate : float
            0~1. 상위 decay_rate 비율만 유지. (0.8 → 하위 20% 제거)
        current_batch : int
            현재 배치 번호.
        decay_freq : int
            decay 적용 주기 (배치 단위).
        min_size : int
            short_head_indices_set 최소 유지 크기.
        grace_period : int
            decay 후 재진입 금지 배치 수. 이 기간 동안 스케치 카운터가
            높아도 get_frequency_percentile에서 제외된다.

        Returns
        -------
        set
            이번 decay에서 저빈도로 내려간 인덱스 집합.
        """
        if current_batch % decay_freq != 0 or current_batch == 0:
            return set()
        if not short_head_indices_set:
            return set()

        # 스케치에서 현재 빈도 맵 구성
        frequent_items = self.sketch.get_frequent_items(
            err_type=frequent_items_error_type.NO_FALSE_POSITIVES,
            threshold=0,
        )
        item_counts = {item[0]: item[1] for item in frequent_items}

        # short_head_indices_set을 빈도 오름차순으로 정렬 (빈도 0 = 최우선 제거)
        sh_sorted = sorted(short_head_indices_set, key=lambda x: item_counts.get(x, 0))

        n_total = len(sh_sorted)
        n_to_remove = int(n_total * (1.0 - decay_rate))
        # min_size 보호: 제거 후 집합 크기가 min_size 이하가 되지 않도록
        n_to_remove = min(n_to_remove, max(0, n_total - min_size))

        if n_to_remove <= 0:
            return set()

        decayed = set(sh_sorted[:n_to_remove])
        short_head_indices_set -= decayed

        # 블랙리스트 등록: grace_period 배치 동안 스케치 재진입 차단
        # → 스케치 카운터가 높아도 즉시 재추가되지 않음
        # → 실질적인 "저빈도로 내려간 것처럼" 동작 → 재전환 유도
        for idx in decayed:
            self._decay_blacklist[idx] = grace_period
        print(f"[CLS-DEBUG] 블랙리스트 등록: {len(decayed)}개 → {grace_period}배치 차단 "
              f"(batch={current_batch})")

        return decayed


def get_indices_and_offsets(selected_rows):
    update_sparse_index_group_batch = selected_rows.reshape(-1)
    non_zero_sparse_index_group_batch = update_sparse_index_group_batch[update_sparse_index_group_batch != 0]
    non_zero_counts = torch.count_nonzero(selected_rows, dim=1)
    cumulative_sum_counts = torch.cumsum(non_zero_counts, dim=0)
    cumulative_sum_counts = cumulative_sum_counts - non_zero_counts
    return non_zero_sparse_index_group_batch, cumulative_sum_counts


def move_nonzero_to_front(tensor):
    sorted_tensor, _ = torch.sort(tensor, dim=1, descending=True)
    return sorted_tensor


def short_head_mask(tensor):
    non_zero_indices = torch.nonzero(tensor, as_tuple=True)
    if non_zero_indices[0].numel() != 0:
        tmp_tensor = torch.cat((torch.zeros((tensor.size(0), 1), dtype=torch.long), tensor[:, 1:]), dim=1)
        updated_tensor = move_nonzero_to_front(tmp_tensor)
        return updated_tensor
    else:
        return tensor


def first_time_operation(x, first_time_code_length, codes_tensors, first_time_tensor, start_first_time_index, num_first_time_indices):
    if start_first_time_index + num_first_time_indices <= first_time_tensor.shape[0]:
        values = first_time_tensor[start_first_time_index:start_first_time_index + num_first_time_indices, :]
        updated_tensors = torch.cat((values[:, :first_time_code_length], codes_tensors[x[:num_first_time_indices], first_time_code_length:]), dim=1)
        remain_tensors = codes_tensors[x[num_first_time_indices:], :]
        result_tensors = torch.cat((updated_tensors, remain_tensors), dim=0)
        return result_tensors
    return codes_tensors[x]


def short_head_operation2(x, short_head_code_length, codes_tensors, short_head_tensor, start_short_head_index, num_first_time_indices, num_short_head_indices, device):
    if start_short_head_index + num_short_head_indices <= short_head_tensor.shape[0]:
        values = short_head_tensor[start_short_head_index:start_short_head_index + num_short_head_indices, :]
        updated_tensors = torch.cat((values[:, :short_head_code_length], torch.zeros((num_short_head_indices, codes_tensors.shape[1] - short_head_code_length), dtype=torch.long, device=device)), dim=1)
        remain_tensors1 = codes_tensors[x[:num_first_time_indices], :]
        remain_tensors2 = codes_tensors[x[(num_first_time_indices + num_short_head_indices):], :]
        result_tensors = torch.cat((remain_tensors1, updated_tensors, remain_tensors2), dim=0)
        return result_tensors
    return codes_tensors[x]


def default_operation(x, codes_tensors):
    return codes_tensors[x]


def add_offsets_to_indices(emb_indices, ln_emb):
    cumulative_sum = torch.cumsum(ln_emb, dim=0)
    offsets = torch.cat((torch.tensor([0]), cumulative_sum[:-1]))
    result_indices = emb_indices + offsets.unsqueeze(1)
    return result_indices, offsets


def remove_offsets_to_indices(original_tensor, chunk_lengths, values_to_add):
    chunks = torch.split(original_tensor, tuple(chunk_lengths))
    values_to_add = values_to_add.view(-1, *((1,) * original_tensor.dim()))
    modified_chunks = list(map(torch.sub, chunks, values_to_add))
    result_tensor = torch.stack(modified_chunks)
    return result_tensor


def get_sliced_indices_and_offsets(sparse_indices, offsets, batch_size, device):
    offsets = torch.cat((offsets, torch.tensor([sparse_indices.shape[0]], device=device)))
    indices = torch.arange(0, len(offsets), batch_size)
    selected_offsets = offsets[indices]
    slice_lengths = selected_offsets[1:] - selected_offsets[:-1]
    sliced_tensors = torch.split(sparse_indices, slice_lengths.tolist())
    offsets_chunk_lengths = indices[1:] - indices[:-1]
    sliced_offsets = remove_offsets_to_indices(offsets[:-1], offsets_chunk_lengths, selected_offsets)
    return sliced_tensors, sliced_offsets


def batch_adaptive_encoding(
    emb_indices_tensor,
    online_frequency_checker,
    freq_tensors,
    codes_tensors,
    first_time_tensor,
    start_first_time_index,
    first_time_code_length,
    short_head_tensor,
    start_short_head_index,
    short_head_code_length,
    selected_ln_emb_offsets,
    device,
    use_short_head,
    iteration_index,
    is_test=False
):
    if emb_indices_tensor.dim() == 1:
        emb_indices_tensor = emb_indices_tensor.unsqueeze(1)
    emb_indices = (emb_indices_tensor + selected_ln_emb_offsets.unsqueeze(1)).view(-1)
    
    online_frequency_checker.add_elements(emb_indices.cpu())

    unique_indices = torch.unique(emb_indices)
    increment_counts = torch.bincount(emb_indices)
    freq_tensors[unique_indices] += increment_counts[unique_indices]

    first_time_mask = freq_tensors[unique_indices] == increment_counts[unique_indices]
    grouped_first_time_indices = torch.nonzero(first_time_mask).view(-1)
    num_first_time_indices = grouped_first_time_indices.shape[0]
    not_first_time_mask = ~first_time_mask
    non_zero_counts = torch.count_nonzero(codes_tensors[unique_indices], dim=1)
    min_iterations = -1

    if use_short_head and iteration_index > min_iterations:
        short_head_long_tail_threshold = 5 if is_test else online_frequency_checker.get_frequency_percentile(0.5)
        short_head_to_update = non_zero_counts != short_head_code_length
        cur_short_head_mask = not_first_time_mask & (freq_tensors[unique_indices] >= short_head_long_tail_threshold)
        short_head_mask = short_head_to_update & cur_short_head_mask & (freq_tensors[unique_indices] != 1)
        grouped_short_head_indices = torch.nonzero(short_head_mask).view(-1)
        num_short_head_indices = grouped_short_head_indices.shape[0]
        default_operation_mask = ~(first_time_mask | short_head_mask)
    else:
        default_operation_mask = ~(first_time_mask)

    if use_short_head and iteration_index > min_iterations and grouped_short_head_indices.numel() != 0:
        short_head_indices = unique_indices[grouped_short_head_indices]

    grouped_default_operation_indices = torch.nonzero(default_operation_mask).view(-1)

    batch_size = unique_indices.shape[0]
    grouped_first_time_mask = torch.zeros(batch_size, dtype=torch.bool, device=device).unsqueeze(1)
    grouped_first_time_mask[:num_first_time_indices] = True
    
    if use_short_head and iteration_index > min_iterations:
        grouped_short_head_mask = torch.zeros(batch_size, dtype=torch.bool, device=device).unsqueeze(1)
        grouped_short_head_mask[num_first_time_indices : num_first_time_indices + num_short_head_indices] = True
        updated_emb_indices = torch.cat([unique_indices[grouped_first_time_indices], unique_indices[grouped_short_head_indices], unique_indices[grouped_default_operation_indices]], dim=0)
        result = torch.where(grouped_first_time_mask,
            first_time_operation(updated_emb_indices, first_time_code_length, codes_tensors, first_time_tensor, start_first_time_index, num_first_time_indices),
            torch.where(grouped_short_head_mask,
                    short_head_operation2(updated_emb_indices, short_head_code_length, codes_tensors, short_head_tensor, start_short_head_index, num_first_time_indices, num_short_head_indices, device),
                    default_operation(updated_emb_indices, codes_tensors)))
        start_short_head_index += torch.sum(short_head_mask).item()
    else:
        updated_emb_indices = torch.cat([unique_indices[grouped_first_time_indices], unique_indices[grouped_default_operation_indices]], dim=0)
        result = torch.where(grouped_first_time_mask,
            first_time_operation(updated_emb_indices, first_time_code_length, codes_tensors, first_time_tensor, start_first_time_index, num_first_time_indices),
            default_operation(updated_emb_indices, codes_tensors))

    codes_tensors[updated_emb_indices] = result

    start_first_time_index += torch.sum(first_time_mask).item()

    sparse_indices, offsets = get_indices_and_offsets(codes_tensors[emb_indices])

    sliced_sparse_indices, sliced_offsets = get_sliced_indices_and_offsets(sparse_indices, offsets, emb_indices_tensor[0].shape[0], device)

    return sliced_sparse_indices, sliced_offsets, start_first_time_index, start_short_head_index


def short_head_operation(x, short_head_hashing):
    short_head_tensors = short_head_hashing.get_hash_embedding_tensors(x)
    return short_head_tensors


def long_tail_operation(x, long_tail_hashing):
    long_tail_tensors = long_tail_hashing.get_hash_embedding_tensors(x)
    # Long-tail indices must stay within this HashEmbedding's shared pool (num_buckets / bins).
    # Do not use num_embeddings here — that is vocabulary size K, not the embedding table width.
    if long_tail_hashing is not None:
        upper = max(int(long_tail_hashing.bins) - 1, 0)
        long_tail_tensors = long_tail_tensors.clamp(0, upper)
    return long_tail_tensors


def batch_adaptive_encoding_with_hashing(
    emb_indices_tensor,
    online_frequency_checker,
    selected_ln_emb_offsets,
    device,
    long_tail_hashing,
    short_head_hashing,
    iteration_index,
    frequency_percentile,
    short_head_indices_set=set(),
    transfer_module=None,
    surge_long_tail_hash=None,
):
    if emb_indices_tensor.dim() == 1:
        emb_indices_tensor = emb_indices_tensor.unsqueeze(1)
    emb_indices = (emb_indices_tensor + selected_ln_emb_offsets.unsqueeze(1)).view(-1)
    online_frequency_checker.add_elements(emb_indices.cpu())

    # 직전 배치에서 전환된 특성 집합 (surge-k 해싱 대상)
    # update()가 호출되기 전에 읽어야 이전 배치의 값을 사용할 수 있음
    prev_newly_frequent = (
        transfer_module.newly_frequent
        if transfer_module is not None
        else set()
    )

    # Here we assume num_hashes in short_head_hashing and long_tail_hashing are the same. So the output dimensions are also the same. No need for zero paddings.
    if short_head_hashing != None:
        short_head_long_tail_threshold, short_head_indices = online_frequency_checker.get_frequency_percentile(frequency_percentile, short_head_indices_set)
        short_head_tensor = torch.tensor(short_head_indices, device=device)
        short_head_mask = torch.any(emb_indices.unsqueeze(1) == short_head_tensor.unsqueeze(0), dim=1)

        # 기본 lt/sh 라우팅 (num_hashes, n)
        num_hashes = long_tail_hashing.num_hashes
        n = emb_indices.shape[0]
        base_lt = long_tail_operation(emb_indices, long_tail_hashing)   # [num_hashes, n]
        base_sh = short_head_operation(emb_indices, short_head_hashing) # [num_hashes, n]
        result = torch.where(short_head_mask, base_sh, base_lt)         # [num_hashes, n]

        # surge-k: 직전 배치에서 전환된 특성을 더 많은 해시 함수로 재해싱
        if surge_long_tail_hash is not None and len(prev_newly_frequent) > 0:
            surge_k = surge_long_tail_hash.num_hashes
            surging_tensor = torch.tensor(
                list(prev_newly_frequent), dtype=torch.long, device=device
            )
            surging_mask = torch.isin(emb_indices, surging_tensor)  # [n]

            if surging_mask.any():
                # 전환 특성 → surge_lt_hash (k=surge_k) 로 해싱, 나머지는 0
                surge_result = surge_long_tail_hash.get_hash_embedding_tensors(
                    emb_indices
                )  # [surge_k, n]
                surge_result = surge_result * surging_mask.long()  # 비전환 위치 → 0

                # 기본 result에서 전환 특성 위치를 0으로 비우고 패딩
                normal_result = result * (~surging_mask).long()  # [num_hashes, n]
                if surge_k > num_hashes:
                    pad = torch.zeros(
                        surge_k - num_hashes, n,
                        dtype=normal_result.dtype, device=device,
                    )
                    normal_result = torch.cat([normal_result, pad], dim=0)  # [surge_k, n]

                # 합산: 전환 특성=surge_result, 나머지=normal_result (0 필터링은 get_indices_and_offsets에서 처리)
                result = normal_result + surge_result  # [surge_k, n]

                n_surging = surging_mask.sum().item()
                print(
                    f"[CLS] 전환 특성 {n_surging}개 → k={surge_k}로 해싱 (충돌 감소)"
                )

        # CLS: 전환 감지 및 임베딩 이전 (short_head_indices_set은 get_frequency_percentile에서 갱신됨)
        if transfer_module is not None:
            # 블랙리스트 디버깅: 만료 직후 배치에서 current/prev 크기 확인
            if (online_frequency_checker._decay_blacklist is not None
                    and len(online_frequency_checker._decay_blacklist) == 0
                    and iteration_index > 0):
                prev_size = len(transfer_module._prev_short_head_set)
                cur_size = len(short_head_indices_set)
                diff = len(short_head_indices_set - transfer_module._prev_short_head_set)
                if diff > 10:  # 의미있는 차이가 있을 때만 출력
                    print(f"[CLS-DEBUG] batch={iteration_index} "
                          f"current={cur_size} prev={prev_size} diff={diff}")
            newly_frequent = transfer_module.update(short_head_indices_set, device, iteration_index)
            if len(newly_frequent) > 0:
                print(f"[CLS] 전환된 특성 수: {len(newly_frequent)}개 → 정보 이전 완료 (alpha={transfer_module.alpha:.2f})")
    else:
        result = long_tail_operation(emb_indices, long_tail_hashing)

    sparse_indices, offsets = get_indices_and_offsets(result.T)

    sliced_sparse_indices, sliced_offsets = get_sliced_indices_and_offsets(sparse_indices, offsets, emb_indices_tensor[0].shape[0], device)

    return sliced_sparse_indices, sliced_offsets


def test_move_nonzero_to_front():
    tensor = torch.tensor([[1,2,0,0], [2,3,4,5], [3,0,0,0]])
    print(f"original tensor: {tensor}")
    sorted_tensor = move_nonzero_to_front(tensor)
    print(f"sorted_tensor: {sorted_tensor}")


def test_remove_offsets_to_indices():
    original_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    chunk_lengths = torch.tensor([3, 3, 3])
    values_to_add = torch.tensor([10, 20, 30])
    result_tensor = remove_offsets_to_indices(original_tensor, chunk_lengths, values_to_add)
    print(result_tensor)


def test_add_offsets_to_indices():
    emb_indices = torch.tensor([[1,1,2,5,8], [10,11,11,23,25], [1,5,6,5,4]])
    ln_emb = torch.tensor([10, 50, 100])
    result_indices = add_offsets_to_indices(emb_indices, ln_emb)
    print(f"result_indices: {result_indices}")


def test_get_sliced_indices_and_offsets():
    sparse_indices = torch.tensor([ 2,  8, 11, 12, 15, 17, 20, 21, 23, 25,  2,  8, 11, 12, 15, 17, 20, 21,
        23, 25,  2,  8, 11, 12, 15, 17, 20, 21, 23, 25,  3,  4,  6,  7,  9, 13,
        14, 19, 24, 25,  1,  2,  5,  6,  9, 12, 16, 20, 23, 25,  2,  4,  7, 11,
        13, 15, 16, 19, 20, 21])

    offsets = torch.tensor([ 0, 10, 20, 30, 40, 50])
    batch_size = 3

    sliced_indices, sliced_offsets = get_sliced_indices_and_offsets(sparse_indices, offsets, batch_size)
    print(f"sliced_indices: {sliced_indices}")
    print(f"sliced_offsets: {sliced_offsets}")


def test_batch_adaptive_encoding():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_indices = torch.tensor([[1,1,1], [1,2,3]], device=device)
    selected_ln_emb_offsets = torch.tensor([0,10], device=device)

    max_index = 10_000
    online_frequency_checker = OnlineFrequencyChecker()
    freq_tensors = torch.zeros(max_index, dtype=torch.long, device=device)
    code_length = 10
    codes_tensors = torch.zeros((max_index, code_length), dtype=torch.long, device=device)
    
    first_time_length = 10
    assert code_length >= first_time_length, "max code_length must >= first_time_length"
    first_time_list = list(combinations(range(1, 26), first_time_length))
    random.shuffle(first_time_list)
    first_time_tensor = torch.tensor(first_time_list, device=device)
    start_first_time_index = 0

    short_head_length = 5
    short_head_list = list(combinations(range(26, 51), short_head_length))
    random.shuffle(short_head_list)
    short_head_tensor = torch.tensor(short_head_list, device=device)
    start_short_head_index = 0

    start_time = time.time()
    sparse_indices1, offsets1, start_first_time_index1, start_short_head_index1 = batch_adaptive_encoding(
        emb_indices, online_frequency_checker, freq_tensors, codes_tensors, first_time_tensor, start_first_time_index,
        first_time_length, short_head_tensor, start_short_head_index, short_head_length, selected_ln_emb_offsets, device, True, True)
    end_time = time.time()
    print(f"encoding sparse_indices1 takes {(end_time - start_time) * 1000}ms")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors emb_indices: {emb_indices}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors sparse_indices1: {sparse_indices1}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors offsets1: {offsets1}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors start_first_time_index1: {start_first_time_index1}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors start_short_head_index1: {start_short_head_index1}")


def test_batch_adaptive_encoding_long_tail_to_short_head_migration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_indices = torch.tensor([[1,1,1], [1,2,3]], device=device)
    selected_ln_emb_offsets = torch.tensor([0,10], device=device)

    max_index = 10_000
    online_frequency_checker = OnlineFrequencyChecker()
    freq_tensors = torch.zeros(max_index, dtype=torch.long, device=device)
    code_length = 10
    codes_tensors = torch.zeros((max_index, code_length), dtype=torch.long, device=device)
    
    first_time_length = 10
    assert code_length >= first_time_length, "max code_length must >= first_time_length"
    max_num_rows = 687
    long_tail_memory_ratio = 0.95
    num_rand_rows_long_tail = int(max_num_rows * long_tail_memory_ratio)
    first_time_list = list(combinations(range(1, num_rand_rows_long_tail), first_time_length))
    random.shuffle(first_time_list)
    first_time_tensor = torch.tensor(first_time_list, device=device)
    start_first_time_index = 0

    short_head_length = 9
    short_head_list = list(combinations(range(num_rand_rows_long_tail, max_num_rows), short_head_length))
    random.shuffle(short_head_list)
    short_head_tensor = torch.tensor(short_head_list, device=device)
    start_short_head_index = 0

    start_time = time.time()
    sparse_indices1, offsets1, start_first_time_index1, start_short_head_index1 = batch_adaptive_encoding(
        emb_indices, online_frequency_checker, freq_tensors, codes_tensors, first_time_tensor, start_first_time_index,
        first_time_length, short_head_tensor, start_short_head_index, short_head_length, selected_ln_emb_offsets, device, True, 2001, False)
    end_time = time.time()
    print(f"encoding sparse_indices1 takes {(end_time - start_time) * 1000}ms")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors emb_indices: {emb_indices}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors sparse_indices1: {sparse_indices1}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors offsets1: {offsets1}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors start_first_time_index1: {start_first_time_index1}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors start_short_head_index1: {start_short_head_index1}")

    emb_indices = torch.tensor([[1,1,2,3], [2,2,2,2]], device=device)
    sparse_indices2, offsets2, start_first_time_index2, start_short_head_index2 = batch_adaptive_encoding(
        emb_indices, online_frequency_checker, freq_tensors, codes_tensors, first_time_tensor, start_first_time_index1,
        first_time_length, short_head_tensor, start_short_head_index1, short_head_length, selected_ln_emb_offsets, device, True, 2002, False)
    print(f"test_batch_adaptive_encoding_optimize_for_tensors emb_indices: {emb_indices}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors sparse_indices2: {sparse_indices2}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors offsets2: {offsets2}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors start_first_time_index2: {start_first_time_index2}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors start_short_head_index2: {start_short_head_index2}")


    emb_indices = torch.tensor([[2,2,3,3], [1,3,4,4]], device=device)
    sparse_indices3, offsets3, start_first_time_index3, start_short_head_index3 = batch_adaptive_encoding(
        emb_indices, online_frequency_checker, freq_tensors, codes_tensors, first_time_tensor, start_first_time_index2,
        first_time_length, short_head_tensor, start_short_head_index2, short_head_length, selected_ln_emb_offsets, device, True, 2003, False)
    print(f"test_batch_adaptive_encoding_optimize_for_tensors emb_indices: {emb_indices}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors sparse_indices3: {sparse_indices3}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors offsets3: {offsets3}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors start_first_time_index3: {start_first_time_index3}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors start_short_head_index3: {start_short_head_index3}")


def test_batch_adaptive_encoding_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_ln_emb_offsets = torch.tensor([0,100,200,300,400,500,600,700,800,900], device=device)

    max_index = 10_000_000
    online_frequency_checker = OnlineFrequencyChecker()
    freq_tensors = torch.zeros(max_index, dtype=torch.long, device=device)
    code_length = 10
    codes_tensors = torch.zeros((max_index, code_length), dtype=torch.long, device=device)
    
    first_time_length = 10
    assert code_length >= first_time_length, "max code_length must >= first_time_length"
    first_time_list = list(combinations(range(1, 26), first_time_length))
    random.shuffle(first_time_list)
    first_time_tensor = torch.tensor(first_time_list, device=device)
    start_first_time_index = 0

    short_head_length = 5
    short_head_list = list(combinations(range(26, 51), short_head_length))
    random.shuffle(short_head_list)
    short_head_tensor = torch.tensor(short_head_list, device=device)
    start_short_head_index = 0
    num_iters = 1000

    start_time = time.time()
    for _ in range(num_iters):
        emb_indices = torch.randint(99, size=(10, 128), device=device)
        sparse_indices1, offsets1, start_first_time_index1, start_short_head_index1 = batch_adaptive_encoding(emb_indices, online_frequency_checker, freq_tensors, codes_tensors, first_time_tensor, start_first_time_index, first_time_length, short_head_tensor, start_short_head_index, short_head_length, selected_ln_emb_offsets, device, True, True)
        start_first_time_index = start_first_time_index1
        start_short_head_index = start_short_head_index1
    end_time = time.time()
    duration = (end_time - start_time) * 1000
    print(f"encoding {num_iters} iterations takes {duration}ms avg {duration / num_iters}")


def test_batch_adaptive_encoding_optimize0_all_operations_performance_testing():
    max_index = 1_000_00
    emb_indices = torch.randint(max_index - 1, size=(128,))
    online_frequency_checker = [OnlineFrequencyChecker() for _ in range(26)]
    k = 0
    freq_tensors = [torch.zeros(max_index, dtype=torch.long) for _ in range(26)]
    first_time_length = 20
    short_head_length = 5
    codes_tensors = [torch.zeros((max_index, first_time_length), dtype=torch.long) for _ in range(26)]
    num_rand_rows = 500

    first_time_list = list(map(lambda _: random.sample(range(1, num_rand_rows + 1), first_time_length), range(max_index + 1)))
    short_head_list = list(map(lambda _: random.sample(range(num_rand_rows + 1, num_rand_rows * 2 + 1), short_head_length), range(max_index + 1)))
    first_time_tensor = torch.tensor(first_time_list)
    short_head_tensor = torch.tensor(short_head_list)
    start_first_time_index = 0
    start_short_head_index = 0
    num_iters = 100000

    start_time = time.time()
    for i in range(num_iters):  
        if i > num_iters / 2:
            emb_indices = torch.randint(max_index - 1, size=(128,))
        else:
            if i == num_iters / 2:
                half_end_time = time.time()
                print(f"first {num_iters // 2} takes {(half_end_time - start_time) * 1000}ms")
                start_time = time.time()
            emb_indices_ones = torch.ones((1, 64), dtype=torch.long)
            emb_indices_twos = torch.full((1, 64), 2, dtype=torch.long)
            emb_indices = torch.cat((emb_indices_ones, emb_indices_twos), dim=1)
            emb_indices = emb_indices.view(-1)

        sparse_indices1, offsets1, start_first_time_index1, start_short_head_index1 = batch_adaptive_encoding_optimize0(emb_indices, online_frequency_checker, k, freq_tensors, codes_tensors, first_time_tensor, start_first_time_index, first_time_length, short_head_tensor, start_short_head_index, short_head_length, True, False)
        start_first_time_index = start_first_time_index1
        start_short_head_index = start_short_head_index1
    
    end_time = time.time()
    duration_in_mills = (end_time - start_time) * 1000
    print(f"encoding second {num_iters // 2} batches takes {duration_in_mills}ms, average {duration_in_mills / num_iters * 2}ms")


def test_batch_adaptive_encoding_with_hashing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_indices = torch.tensor([[1,1,1], [1,2,3]], device=device)
    num_embeddings = 100
    selected_ln_emb_offsets = torch.tensor([0, num_embeddings], device=device)
    max_index = 10_000
    online_frequency_checker = OnlineFrequencyChecker()
    embedding_dimension = 16
    num_buckets = 25
    num_hashes = 3
    long_tail_hash = HashEmbedding(num_embeddings, embedding_dimension, num_buckets=num_buckets, num_hashes=num_hashes, append_weight=False, device=device)
    short_head_hash = HashEmbedding(num_embeddings, embedding_dimension, num_buckets=num_buckets, num_hashes=num_hashes, append_weight=False, offsets=num_buckets, device=device)

    start_time = time.time()
    sparse_indices1, offsets1 = batch_adaptive_encoding_with_hashing(
        emb_indices, online_frequency_checker, selected_ln_emb_offsets, device, long_tail_hash, short_head_hash, -1, 0.5)
    end_time = time.time()
    print(f"encoding sparse_indices1 takes {(end_time - start_time) * 1000}ms")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors emb_indices: {emb_indices}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors sparse_indices1: {sparse_indices1}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors offsets1: {offsets1}")


def test_batch_adaptive_encoding_with_hashing_long_tail_to_short_head_migration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_indices = torch.tensor([[1,1,1], [1,2,3]], device=device)
    num_embeddings = 100
    selected_ln_emb_offsets = torch.tensor([0, num_embeddings], device=device)
    max_index = 10_000
    online_frequency_checker = OnlineFrequencyChecker()
    embedding_dimension = 16
    capacity = 687
    long_tail_memory_ratio = 0.95
    long_tail_num_hashes = 10
    short_head_num_hashes = 10
    num_buckets_long_tail = int(capacity * long_tail_memory_ratio)
    num_buckets_short_head = capacity - num_buckets_long_tail
    long_tail_hash = HashEmbedding(num_embeddings, embedding_dimension, num_buckets=num_buckets_long_tail, num_hashes=long_tail_num_hashes, append_weight=False, device=device)
    short_head_hash = HashEmbedding(num_embeddings, embedding_dimension, num_buckets=num_buckets_short_head, num_hashes=short_head_num_hashes, append_weight=False, offsets=num_buckets_long_tail, device=device)
    short_head_indices = set()

    sparse_indices1, offsets1 = batch_adaptive_encoding_with_hashing(
        emb_indices, online_frequency_checker, selected_ln_emb_offsets, device, long_tail_hash, short_head_hash, -1, 0.5, short_head_indices)
    print(f"test_batch_adaptive_encoding_optimize_for_tensors emb_indices: {emb_indices}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors sparse_indices1: {sparse_indices1}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors offsets1: {offsets1}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors short_head_indices: {short_head_indices}")

    emb_indices = torch.tensor([[1,1,5,6], [1,1,1,1]], device=device)
    sparse_indices2, offsets2 = batch_adaptive_encoding_with_hashing(
        emb_indices, online_frequency_checker, selected_ln_emb_offsets, device, long_tail_hash, short_head_hash, -1, 0.5, short_head_indices)
    print(f"test_batch_adaptive_encoding_optimize_for_tensors emb_indices: {emb_indices}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors sparse_indices2: {sparse_indices2}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors offsets2: {offsets2}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors short_head_indices: {short_head_indices}")


    emb_indices = torch.tensor([[1,1,5,6], [2,3,4,5]], device=device)
    sparse_indices3, offsets3 = batch_adaptive_encoding_with_hashing(
        emb_indices, online_frequency_checker, selected_ln_emb_offsets, device, long_tail_hash, short_head_hash, -1, 0.5, short_head_indices)
    print(f"test_batch_adaptive_encoding_optimize_for_tensors emb_indices: {emb_indices}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors sparse_indices3: {sparse_indices3}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors offsets3: {offsets3}")
    print(f"test_batch_adaptive_encoding_optimize_for_tensors short_head_indices: {short_head_indices}")


if __name__ == "__main__":
    # test_batch_adaptive_encoding_performance()
    # test_batch_adaptive_encoding()
    # test_batch_adaptive_encoding_long_tail_to_short_head_migration()
    # test_batch_adaptive_encoding_with_hashing()
    test_batch_adaptive_encoding_with_hashing_long_tail_to_short_head_migration()
