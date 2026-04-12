from __future__ import absolute_import, division, print_function, unicode_literals

import bisect
import collections
import sys
from collections import deque

# numpy
import numpy as np
import math
import os

# pytorch
import torch
from numpy import random as ra
from torch.utils.data import Dataset
import dataset_helper
import warnings

class KDD12Dataset:
    def __init__(
        self,
        count,
        data_cat,
        data_int,
        data_T,
        split,
        hash_flag,
        hc_flag,
        compress_rate,
        hot_features=None,
        hash_rate=0.5,
    ):
        self.hash_rate = hash_rate
        self.hash_flag = hash_flag
        self.hc_flag = hc_flag
        self.compress_rate = compress_rate
        self.data_cat = data_cat[split]
        self.data_T = data_T[split]
        self.hash_size = np.zeros(11)
        self.hot_features = hot_features
        self.counts = np.array(count, dtype=np.int32)
        self.sum_count = np.zeros(11, dtype=np.int32)
        for i in range(1, 11):
            self.sum_count[i] = self.counts[i-1] + self.sum_count[i-1]

        #print(f"sum: {self.sum_count}")
        if self.hash_flag:
            for i in range(11):
                if self.counts[i] > 2000 * self.compress_rate:
                    self.counts[i] = int(
                        math.ceil(self.counts[i] * self.compress_rate))
                    self.hash_size[i] = self.counts[i]
        if self.hc_flag:
            for i in range(11):
                if self.counts[i] > 2000 * self.compress_rate:
                    self.hash_size[i] = int(
                        round(self.counts[i] * self.compress_rate * self.hash_rate + 0.55))
                    self.counts[i] = int(
                        self.hash_size[i] + len(self.hot_features[i]))
        print(f"count: {self.counts}, hash_size: {self.hash_size}")

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]
        data_c = np.array(self.data_cat[index])
        data_c -= self.sum_count
        # print(f"data_c: {data_c}, {self.sum_count}")
        if self.hash_flag or self.hc_flag:
            for i in range(11):
                if self.hash_flag and self.hash_size[i] != 0:
                    data_c[i] = data_c[i] % self.hash_size[i]
                elif self.hc_flag and self.hash_size[i] != 0:
                    if data_c[i] not in self.hot_features[i]:
                        data_c[i] = len(self.hot_features[i]) + \
                            data_c[i] % self.hash_size[i]
                    else:
                        data_c[i] = self.hot_features[i][data_c[i]]
        
        return (data_c, None, self.data_T[index])

    def __len__(self):
        return len(self.data_T)


class AvazuDataset:
    def __init__(
        self,
        count,
        data_cat,
        data_int,
        data_T,
        split,
        hash_flag,
        hc_flag,
        compress_rate,
        hot_features=None,
        hash_rate=0.5,
    ):
        self.hash_rate = hash_rate
        self.hash_flag = hash_flag
        self.hc_flag = hc_flag
        self.compress_rate = compress_rate
        if (split == 'train'):
            train_len = 40428967 - 4218938
            self.data_cat = data_cat[:train_len]
            self.data_T = data_T[:train_len]
        if (split == 'test'):
            train_len = 40428967 - 4218938
            self.data_cat = data_cat[train_len:]
            self.data_T = data_T[train_len:]
        self.hash_size = np.zeros(22)
        self.hot_features = hot_features
        self.counts = np.array(count, dtype=np.int32)
        self.sum_count = np.zeros(22, dtype=np.int32)
        for i in range(1, 22):
            self.sum_count[i] = self.counts[i-1] + self.sum_count[i-1]
        #print(f"sum: {self.sum_count}")
        if self.hash_flag:
            for i in range(22):
                if self.counts[i] > 2000 * self.compress_rate:
                    self.counts[i] = int(
                        math.ceil(self.counts[i] * self.compress_rate))
                    self.hash_size[i] = self.counts[i]
        if self.hc_flag:
            for i in range(22):
                if self.counts[i] > 2000 * self.compress_rate:
                    self.hash_size[i] = int(
                        round(self.counts[i] * self.compress_rate * self.hash_rate + 0.55))
                    self.counts[i] = int(
                        self.hash_size[i] + len(self.hot_features[i]))
        print(f"count: {self.counts}, hash_size: {self.hash_size}, len: {len(self.data_T)}")

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]
        data_c = np.array(self.data_cat[index])
        #print(f"data_c: {data_c}, {self.sum_count}")
        data_c -= self.sum_count
        if self.hash_flag or self.hc_flag:
            for i in range(22):
                if self.hash_flag and self.hash_size[i] != 0:
                    data_c[i] = data_c[i] % self.hash_size[i]
                elif self.hc_flag and self.hash_size[i] != 0:
                    if data_c[i] not in self.hot_features[i]:
                        data_c[i] = len(self.hot_features[i]) + \
                            data_c[i] % self.hash_size[i]
                    else:
                        data_c[i] = self.hot_features[i][data_c[i]]
        
        return (data_c, None, self.data_T[index])

    def __len__(self):
        return len(self.data_T)

class CriteotbDataSet:
    def __init__(
        self,
        split,
        hash_flag,
        compress_rate,
        batch_size,
        max_ind_range=-1,
        raw_path="",
        pro_data=""
    ):
        super(CriteotbDataSet, self).__init__()
        self.hash_flag = hash_flag
        self.compress_rate = compress_rate
        self.hash_size = np.zeros(26, dtype=np.int32)
        self.tar_fea = 1  # single target
        self.den_fea = 13  # 13 dense  features
        self.spa_fea = 26  # 26 sparse features
        self.max_ind_range = max_ind_range
        self.split = split

        days = 24
        out_file = "terabyte_processed"
        lstr = raw_path.split("/")
        self.d_path = "/".join(lstr[0:-1]) + "/"
        self.d_file = lstr[-1]

        total_file = self.d_path + self.d_file + "_day_count.npz"
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"]
        # compute offsets per file
        self.offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(days):
            self.offset_per_file[i + 1] += self.offset_per_file[i]

        with np.load(pro_data) as data:
            X_int = data["X_int"]  # continuous  feature
            X_cat = data["X_cat"]  # categorical feature
            y = data["y"]          # target
            self.counts = data["counts"]

        print(f"counts: {self.counts}")
        self.index = 0
        self.batch_size = batch_size

        if self.hash_flag:
            for i in range(26):
                if self.counts[i] > 2000 * self.compress_rate:
                    self.counts[i] = int(
                        math.ceil(self.counts[i] * self.compress_rate))
                self.hash_size[i] = self.counts[i]

        # create reordering
        indices = np.arange(len(y))
        indices = np.array_split(indices, self.offset_per_file[1:-1])

        train_indices = np.concatenate(indices[:-1])
        test_indices = indices[-1]
        test_indices, val_indices = np.array_split(test_indices, 2)

        print("Defined %s indices..." % (split))

        # randomize train data (across days)
        train_indices = np.random.permutation(train_indices)
        print("Randomized indices across days ...")

        if split == 'train':
            self.X_int = [X_int[i] for i in train_indices]
            self.X_cat = [X_cat[i] for i in train_indices]
            self.y = [y[i] for i in train_indices]
        elif split == 'val':
            self.X_int = [X_int[i] for i in val_indices]
            self.X_cat = [X_cat[i] for i in val_indices]
            self.y = [y[i] for i in val_indices]
        elif split == 'test':
            self.X_int = [X_int[i] for i in test_indices]
            self.X_cat = [X_cat[i] for i in test_indices]
            self.y = [y[i] for i in test_indices]

        print("Split data according to indices...")


    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]
        data_c = np.array(self.X_cat[index])
        if self.hash_flag:
            for i in range(26):
                if self.hash_size[i] != 0:
                    data_c[i] = data_c[i] % self.hash_size[i]
        
        return (data_c, self.X_int[index], self.y[index])
        
        # if self.max_ind_range > 0:
        #     return self.X_cat[index] % self.max_ind_range, self.X_int[index], self.y[index]
        # else:
        #     return self.X_cat[index], self.X_int[index], self.y[index]


class KaggleDataset:
    def __init__(
        self,
        count,
        data_cat,
        data_int,
        data_T,
        split,
        hash_flag,
        hc_flag,
        compress_rate,
        hot_features=None,
        hash_rate=0.5,
    ):
        self.hash_rate = hash_rate
        self.hash_flag = hash_flag
        self.hc_flag = hc_flag
        self.compress_rate = compress_rate
        if (split == 'train'):
            # special designed dataset
            # arr = np.arange(45840617)
            # result = np.array_split(arr, 7)
            # result = np.concatenate((result[0], result[2], result[4]))
            # self.data_cat = data_cat[result]
            # self.data_int = data_int[result]
            # self.data_T = data_T[result]
            # print(f"len: {len(result)}")
            train_len = 45840617 * 6 // 7
            self.data_cat = data_cat[:train_len]
            self.data_int = data_int[:train_len]
            self.data_T = data_T[:train_len]
        if (split == 'test'):
            train_len = 45840617 * 6 // 7
            self.data_cat = data_cat[train_len:]
            self.data_int = data_int[train_len:]
            self.data_T = data_T[train_len:]
        self.hash_size = np.zeros(26, dtype=np.int32)
        self.hot_features = hot_features
        self.counts = np.array(count, dtype=np.int32)
        if self.hash_flag:
            for i in range(26):
                if self.counts[i] > 2000 * self.compress_rate:
                    self.counts[i] = int(
                        math.ceil(self.counts[i] * self.compress_rate))
                self.hash_size[i] = self.counts[i]
        if self.hc_flag:
            for i in range(26):
                if self.counts[i] > 2000 * self.compress_rate:
                    self.hash_size[i] = int(
                        round(self.counts[i] * self.compress_rate * self.hash_rate + 0.55))
                    self.counts[i] = int(
                        self.hash_size[i] + len(self.hot_features[i]))
        print(f"count: {self.counts}, hash_size: {self.hash_size}")

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]
        data_c = np.array(self.data_cat[index])
        if self.hash_flag or self.hc_flag:
            # data_c %= self.hash_size
            for i in range(26):
                if self.hash_flag and self.hash_size[i] != 0:
                    data_c[i] = data_c[i] % self.hash_size[i]
                elif self.hc_flag and self.hash_size[i] != 0:
                    if data_c[i] not in self.hot_features[i]:
                        data_c[i] = len(self.hot_features[i]) + \
                            data_c[i] % self.hash_size[i]
                    else:
                        data_c[i] = self.hot_features[i][data_c[i]]
        return (data_c, self.data_int[index], self.data_T[index])

    def __len__(self):
        return len(self.data_T)


def collate_wrapper_criteo_offset3(list_of_tuples):
    transposed_data = list(zip(*list_of_tuples))
    #X_int = torch.tensor(transposed_data[1], dtype=torch.float)
    X_cat = torch.tensor(np.array(transposed_data[0]), dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

    return None, torch.stack(lS_o), torch.stack(lS_i), T

def collate_wrapper_criteo_offset2(list_of_tuples):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        transposed_data = list(zip(*list_of_tuples))
        # int_array = np.array(transposed_data[1])
        # X_int = torch.log(torch.tensor(int_array, dtype=torch.float) + 1)
        X_int = torch.log(torch.tensor(transposed_data[1], dtype=torch.float) + 1)
        # X_int = torch.tensor(np.array(transposed_data[1]), dtype=torch.float)
        
        # X_cat = torch.tensor(np.array(transposed_data[0]), dtype=torch.long)
        # cat_array = np.array(transposed_data[0])
        # X_cat = torch.tensor(cat_array, dtype=torch.long)
        X_cat = torch.tensor(transposed_data[0], dtype=torch.long)
        
        # T_array = np.array(transposed_data[2])
        # T = torch.tensor(T_array, dtype=torch.float32).view(-1, 1)
        T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

        batchSize = X_cat.shape[0]
        featureCnt = X_cat.shape[1]

        lS_i = [X_cat[:, i] for i in range(featureCnt)]
        lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

        return X_int, torch.stack(lS_o), torch.stack(lS_i), T


def collate_wrapper_criteo_offset(list_of_tuples):
    transposed_data = list_of_tuples[0]
    X_int = transposed_data[1]
    X_cat = torch.from_numpy(transposed_data[0])
    X_int = torch.tensor(transposed_data[1], dtype=torch.float)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
    # print(f"X_int: {X_int}, lS_o: {lS_o}, lS_i: {lS_i}, T: {T.shape}")
    return X_int, torch.stack(lS_o), torch.stack(lS_i), T


# Conversion from offset to length
def offset_to_length_converter(lS_o, lS_i):
    def diff(tensor):
        return tensor[1:] - tensor[:-1]

    return torch.stack(
        [
            diff(torch.cat((S_o, torch.tensor(lS_i[ind].shape))).int())
            for ind, S_o in enumerate(lS_o)
        ]
    )


def collate_wrapper_criteo_length(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = torch.stack([X_cat[:, i] for i in range(featureCnt)])
    lS_o = torch.stack([torch.tensor(range(batchSize))
                       for _ in range(featureCnt)])

    lS_l = offset_to_length_converter(lS_o, lS_i)

    return X_int, lS_l, lS_i, T


def calc_bucket_hot(data, threshold, compress_rate, hash_rate, count):
    print(data.shape)
    hot_dict = []
    tot = 0
    unique_values = []
    counts = []
    for i in range(26):
        hot_dict.append({})
        if count[i] > 200:
            uni, cnt = np.unique(data[:, i], return_counts=True)
            uni += tot
            unique_values.extend(uni.tolist())
            counts.extend(cnt.tolist())
            tot += count[i]

    hot_nums = int(tot * compress_rate * (1.0 - hash_rate))
    print(f"hot_nums: {hot_nums}")
    idx = np.argsort(np.array(counts))[-hot_nums:]
    unique_values = np.array(unique_values)[idx]
    unique_values.sort()
    lst = 0
    for i in range(26):
        if count[i] > 200:
            tmp = 0
            while lst < hot_nums and unique_values[lst] < count[i]:
                hot_dict[i][unique_values[lst]] = tmp
                lst += 1
                tmp += 1
            unique_values -= count[i]

    return hot_dict


def make_criteo_data_and_loaders(args, offset_to_length_converter=False):
    if args.data_set == 'kdd12':
        data_cat = np.memmap(args.cat_path, dtype=np.int32,
                             mode='r', shape=(149639105, 11))
        data_T = np.memmap(args.label_path, dtype=np.int32,
                           mode='r', shape=(149639105,))
        data_int = None
        data_count = np.memmap(
            args.count_path, dtype=np.int32, mode='r', shape=(12,))
        hot_features = None

        count = np.array(data_count)
        new_count = np.zeros(11)
        for i in range(11):
            new_count[i] = count[i+1] - count[i]
        print(f"new_count: {new_count}")
        np.random.seed(2023)
        tot_len = 149639105
        index = np.arange(tot_len)
        np.random.shuffle(index)
        print(f"index: {index.shape} {index}")

        test_size = int(0.1 * tot_len)
        train_size = tot_len - test_size
        test, train = index[:test_size], index[test_size:]
        print(f"index: {test.shape} {train.shape}")
        train_data = KDD12Dataset(
            new_count,
            data_cat,
            data_int,
            data_T,
            train,
            args.hash_flag,
            args.bucket_flag,
            args.compress_rate,
            hot_features,
            args.hash_rate,
        )
        test_data = KDD12Dataset(
            new_count,
            data_cat,
            data_int,
            data_T,
            test,
            args.hash_flag,
            args.bucket_flag,
            args.compress_rate,
            hot_features,
            args.hash_rate,
        )

        collate_wrapper_criteo = collate_wrapper_criteo_offset3
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.test_mini_batch_size,
            shuffle=False,
            num_workers=args.test_num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )
    if args.data_set == 'avazu':
        data_cat = np.memmap(args.cat_path, dtype=np.int32,
                             mode='r', shape=(40428967, 22))
        data_T = np.memmap(args.label_path, dtype=np.int32,
                           mode='r', shape=(40428967,))
        data_int = None
        data_count = np.memmap(
            args.count_path, dtype=np.int32, mode='r', shape=(23, ))
        print(f"data_cat shape: {data_cat.shape}")
        print(f"data_T shape: {data_T.shape}")
        print(f"data_count shape: {data_count.shape}")
        print(f"data_count: {data_count}")
        hot_features = None
        count = np.array(data_count)
        new_count = np.zeros(22)
        for i in range(22):
            new_count[i] = count[i+1] - count[i]

        train_data = AvazuDataset(
            new_count,
            data_cat,
            data_int,
            data_T,
            'train',
            args.hash_flag,
            args.bucket_flag,
            args.compress_rate,
            hot_features,
            args.hash_rate,
        )
        test_data = AvazuDataset(
            new_count,
            data_cat,
            data_int,
            data_T,
            'test',
            args.hash_flag,
            args.bucket_flag,
            args.compress_rate,
            hot_features,
            args.hash_rate,
        )

        collate_wrapper_criteo = collate_wrapper_criteo_offset3
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.test_mini_batch_size,
            shuffle=False,
            num_workers=args.test_num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )
    if args.data_set == 'kaggle':

        # data_cat = np.memmap("../criteo/new_sparse.bin", dtype = np.int32, mode = 'r', shape=(45840617,26))
        # data_int = np.memmap("../criteo/kaggle_processed_dense.bin", dtype = np.float32, mode = 'r', shape=(45840617,13))
        # data_T = np.memmap("../criteo/kaggle_processed_label.bin", dtype = np.int32, mode = 'r', shape=(45840617,))
        # data_count = np.memmap("../criteo/origin/kaggle_processed_count.bin", dtype = np.int32, mode = 'r')

        # reuse kaggleAdDisplayChallenge_processed file to generate data_cat, data_int, data_T, and data_count
        # data_cat = np.memmap(args.cat_path, dtype=np.int32,
        #                      mode='r', shape=(45840617, 26))
        # data_int = np.memmap(args.dense_path, dtype=np.float32,
        #                      mode='r', shape=(45840617, 13))
        # data_T = np.memmap(args.label_path, dtype=np.int32,
        #                    mode='r', shape=(45840617,))
        # data_count = np.memmap(
        #     args.count_path, dtype=np.int32, mode='r', shape=(27, ))

        pro_data = "./input/kaggleAdDisplayChallenge_processed.npz"
        data_int, data_cat, data_T, data_count = dataset_helper.get_split_data(pro_data)

        hot_features = None
        new_count = np.array(data_count)
        train_data = KaggleDataset(
            new_count,
            data_cat,
            data_int,
            data_T,
            'train',
            args.hash_flag,
            args.bucket_flag,
            args.compress_rate,
            hot_features,
            args.hash_rate,
        )
        test_data = KaggleDataset(
            new_count,
            data_cat,
            data_int,
            data_T,
            'test',
            args.hash_flag,
            args.bucket_flag,
            args.compress_rate,
            hot_features,
            args.hash_rate,
        )

        collate_wrapper_criteo = collate_wrapper_criteo_offset2
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.test_mini_batch_size,
            shuffle=False,
            num_workers=args.test_num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )
    elif args.data_set == 'terabyte':
        hot_features = None

        train_data = CriteotbDataSet(
            'train',
            args.hash_flag,
            args.compress_rate,
            batch_size=args.mini_batch_size,
            max_ind_range=args.max_ind_range,
            raw_path=args.raw_data_file,
            pro_data=args.processed_data_file
        )
        test_data = CriteotbDataSet(
            'test',
            args.hash_flag,
            args.compress_rate,
            batch_size=args.test_mini_batch_size,
            max_ind_range=args.max_ind_range,
            raw_path=args.raw_data_file,
            pro_data=args.processed_data_file
        )
        collate_wrapper_criteo = collate_wrapper_criteo_offset2

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.test_num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )

    return train_data, train_loader, test_data, test_loader


# uniform ditribution (input data)
class RandomDataset(Dataset):
    def __init__(
        self,
        m_den,
        ln_emb,
        data_size,
        num_batches,
        mini_batch_size,
        num_indices_per_lookup,
        num_indices_per_lookup_fixed,
        num_targets=1,
        round_targets=False,
        data_generation="random",
        trace_file="",
        enable_padding=False,
        reset_seed_on_access=False,
        rand_data_dist="uniform",
        rand_data_min=1,
        rand_data_max=1,
        rand_data_mu=-1,
        rand_data_sigma=1,
        rand_seed=0,
    ):
        # compute batch size
        nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
        if num_batches != 0:
            nbatches = num_batches
            data_size = nbatches * mini_batch_size
            # print("Total number of batches %d" % nbatches)

        # save args (recompute data_size if needed)
        self.m_den = m_den
        self.ln_emb = ln_emb
        self.data_size = data_size
        self.num_batches = nbatches
        self.mini_batch_size = mini_batch_size
        self.num_indices_per_lookup = num_indices_per_lookup
        self.num_indices_per_lookup_fixed = num_indices_per_lookup_fixed
        self.num_targets = num_targets
        self.round_targets = round_targets
        self.data_generation = data_generation
        self.trace_file = trace_file
        self.enable_padding = enable_padding
        self.reset_seed_on_access = reset_seed_on_access
        self.rand_seed = rand_seed
        self.rand_data_dist = rand_data_dist
        self.rand_data_min = rand_data_min
        self.rand_data_max = rand_data_max
        self.rand_data_mu = rand_data_mu
        self.rand_data_sigma = rand_data_sigma

    def reset_numpy_seed(self, numpy_rand_seed):
        np.random.seed(numpy_rand_seed)
        # torch.manual_seed(numpy_rand_seed)

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        # WARNING: reset seed on access to first element
        # (e.g. if same random samples needed across epochs)
        if self.reset_seed_on_access and index == 0:
            self.reset_numpy_seed(self.rand_seed)

        # number of data points in a batch
        n = min(self.mini_batch_size, self.data_size -
                (index * self.mini_batch_size))

        # generate a batch of dense and sparse features
        if self.data_generation == "random":
            (X, lS_o, lS_i) = generate_dist_input_batch(
                self.m_den,
                self.ln_emb,
                n,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed,
                rand_data_dist=self.rand_data_dist,
                rand_data_min=self.rand_data_min,
                rand_data_max=self.rand_data_max,
                rand_data_mu=self.rand_data_mu,
                rand_data_sigma=self.rand_data_sigma,
            )
        elif self.data_generation == "synthetic":
            (X, lS_o, lS_i) = generate_synthetic_input_batch(
                self.m_den,
                self.ln_emb,
                n,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed,
                self.trace_file,
                self.enable_padding,
            )
        else:
            sys.exit(
                "ERROR: --data-generation=" + self.data_generation + " is not supported"
            )

        # generate a batch of target (probability of a click)
        T = generate_random_output_batch(
            n, self.num_targets, self.round_targets)

        return (X, lS_o, lS_i, T)

    def __len__(self):
        # WARNING: note that we produce bacthes of outputs in __getitem__
        # therefore we should use num_batches rather than data_size below
        return self.num_batches


def collate_wrapper_random_offset(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X, torch.stack(lS_o), lS_i, T)


def collate_wrapper_random_length(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X, offset_to_length_converter(torch.stack(lS_o), lS_i), lS_i, T)


def make_random_data_and_loader(
    args,
    ln_emb,
    m_den,
    offset_to_length_converter=False,
):

    train_data = RandomDataset(
        m_den,
        ln_emb,
        args.data_size,
        args.num_batches,
        args.mini_batch_size,
        args.num_indices_per_lookup,
        args.num_indices_per_lookup_fixed,
        1,  # num_targets
        args.round_targets,
        args.data_generation,
        args.data_trace_file,
        args.data_trace_enable_padding,
        reset_seed_on_access=True,
        rand_data_dist=args.rand_data_dist,
        rand_data_min=args.rand_data_min,
        rand_data_max=args.rand_data_max,
        rand_data_mu=args.rand_data_mu,
        rand_data_sigma=args.rand_data_sigma,
        rand_seed=args.numpy_rand_seed,
    )  # WARNING: generates a batch of lookups at once

    test_data = RandomDataset(
        m_den,
        ln_emb,
        args.data_size,
        args.num_batches,
        args.mini_batch_size,
        args.num_indices_per_lookup,
        args.num_indices_per_lookup_fixed,
        1,  # num_targets
        args.round_targets,
        args.data_generation,
        args.data_trace_file,
        args.data_trace_enable_padding,
        reset_seed_on_access=True,
        rand_data_dist=args.rand_data_dist,
        rand_data_min=args.rand_data_min,
        rand_data_max=args.rand_data_max,
        rand_data_mu=args.rand_data_mu,
        rand_data_sigma=args.rand_data_sigma,
        rand_seed=args.numpy_rand_seed,
    )

    collate_wrapper_random = collate_wrapper_random_offset
    if offset_to_length_converter:
        collate_wrapper_random = collate_wrapper_random_length

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )
    return train_data, train_loader, test_data, test_loader


def generate_random_data(
    m_den,
    ln_emb,
    data_size,
    num_batches,
    mini_batch_size,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    num_targets=1,
    round_targets=False,
    data_generation="random",
    trace_file="",
    enable_padding=False,
    length=False,  # length for caffe2 version (except dlrm_s_caffe2)
):
    nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
    if num_batches != 0:
        nbatches = num_batches
        data_size = nbatches * mini_batch_size
    # print("Total number of batches %d" % nbatches)

    # inputs
    lT = []
    lX = []
    lS_offsets = []
    lS_indices = []
    for j in range(0, nbatches):
        # number of data points in a batch
        n = min(mini_batch_size, data_size - (j * mini_batch_size))

        # generate a batch of dense and sparse features
        if data_generation == "random":
            (Xt, lS_emb_offsets, lS_emb_indices) = generate_uniform_input_batch(
                m_den,
                ln_emb,
                n,
                num_indices_per_lookup,
                num_indices_per_lookup_fixed,
                length,
            )
        elif data_generation == "synthetic":
            (Xt, lS_emb_offsets, lS_emb_indices) = generate_synthetic_input_batch(
                m_den,
                ln_emb,
                n,
                num_indices_per_lookup,
                num_indices_per_lookup_fixed,
                trace_file,
                enable_padding,
            )
        else:
            sys.exit(
                "ERROR: --data-generation=" + data_generation + " is not supported"
            )
        # dense feature
        lX.append(Xt)
        # sparse feature (sparse indices)
        lS_offsets.append(lS_emb_offsets)
        lS_indices.append(lS_emb_indices)

        # generate a batch of target (probability of a click)
        P = generate_random_output_batch(n, num_targets, round_targets)
        lT.append(P)

    return (nbatches, lX, lS_offsets, lS_indices, lT)


def generate_random_output_batch(n, num_targets, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(
            np.float32)).astype(np.float32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)

    return torch.tensor(P)


# uniform ditribution (input data)
def generate_uniform_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    length,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
            # sparse indices to be used per embedding
            r = ra.random(sparse_group_size)
            sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int32(sparse_group.size)
            # store lengths and indices
            if length:  # for caffe2 version
                lS_batch_offsets += [sparse_group_size]
            else:
                lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)


# random data from uniform or gaussian ditribution (input data)
def generate_dist_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    rand_data_dist,
    rand_data_min,
    rand_data_max,
    rand_data_mu,
    rand_data_sigma,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
            # sparse indices to be used per embedding
            if rand_data_dist == "gaussian":
                if rand_data_mu == -1:
                    rand_data_mu = (rand_data_max + rand_data_min) / 2.0
                r = ra.normal(rand_data_mu, rand_data_sigma, sparse_group_size)
                sparse_group = np.clip(r, rand_data_min, rand_data_max)
                sparse_group = np.unique(sparse_group).astype(np.int64)
            elif rand_data_dist == "uniform":
                r = ra.random(sparse_group_size)
                sparse_group = np.unique(
                    np.round(r * (size - 1)).astype(np.int64))
            else:
                raise (
                    rand_data_dist,
                    "distribution is not supported. \
                     please select uniform or gaussian",
                )

            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)


# synthetic distribution (input data)
def generate_synthetic_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    trace_file,
    enable_padding=False,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for i, size in enumerate(ln_emb):
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    max(1, np.round(r * min(size, num_indices_per_lookup))[0])
                )
            # sparse indices to be used per embedding
            file_path = trace_file
            line_accesses, list_sd, cumm_sd = read_dist_from_file(
                file_path.replace("j", str(i))
            )
            # debug prints
            # print("input")
            # print(line_accesses); print(list_sd); print(cumm_sd);
            # print(sparse_group_size)
            # approach 1: rand
            # r = trace_generate_rand(
            #     line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            # )
            # approach 2: lru
            r = trace_generate_lru(
                line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            )
            # WARNING: if the distribution in the file is not consistent
            # with embedding table dimensions, below mod guards against out
            # of range access
            sparse_group = np.unique(r).astype(np.int64)
            minsg = np.min(sparse_group)
            maxsg = np.max(sparse_group)
            if (minsg < 0) or (size <= maxsg):
                print(
                    "WARNING: distribution is inconsistent with embedding "
                    + "table size (using mod to recover and continue)"
                )
                sparse_group = np.mod(sparse_group, size).astype(np.int64)
            # sparse_group = np.unique(np.array(np.mod(r, size-1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)


def generate_stack_distance(cumm_val, cumm_dist, max_i, i, enable_padding=False):
    u = ra.rand(1)
    if i < max_i:
        # only generate stack distances up to the number of new references seen so far
        j = bisect.bisect(cumm_val, i) - 1
        fi = cumm_dist[j]
        u *= fi  # shrink distribution support to exclude last values
    elif enable_padding:
        # WARNING: disable generation of new references (once all have been seen)
        fi = cumm_dist[0]
        # remap distribution support to exclude first value
        u = (1.0 - fi) * u + fi

    for (j, f) in enumerate(cumm_dist):
        if u <= f:
            return cumm_val[j]


# WARNING: global define, must be consistent across all synthetic functions
cache_line_size = 1


def trace_generate_lru(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
):
    max_sd = list_sd[-1]
    l = len(line_accesses)
    i = 0
    ztrace = deque()
    for _ in range(out_trace_len):
        sd = generate_stack_distance(
            list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0

        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses[0]
            del line_accesses[0]
            line_accesses.append(line_ref)
            mem_ref = np.uint64(
                line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(
                line_ref * cache_line_size + mem_ref_within_line)
            del line_accesses[l - sd]
            line_accesses.append(line_ref)
        # save generated memory reference
        ztrace.append(mem_ref)

    return ztrace


def trace_generate_rand(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
):
    max_sd = list_sd[-1]
    l = len(line_accesses)  # !!!Unique,
    i = 0
    ztrace = []
    for _ in range(out_trace_len):
        sd = generate_stack_distance(
            list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0
        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses.pop(0)
            line_accesses.append(line_ref)
            mem_ref = np.uint64(
                line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(
                line_ref * cache_line_size + mem_ref_within_line)
        ztrace.append(mem_ref)

    return ztrace


def trace_profile(trace, enable_padding=False):
    # number of elements in the array (assuming 1D)
    # n = trace.size

    rstack = deque()  # S
    stack_distances = deque()  # SDS
    line_accesses = deque()  # L
    for x in trace:
        r = np.uint64(x / cache_line_size)
        l = len(rstack)
        try:  # found #
            i = rstack.index(r)
            # WARNING: I believe below is the correct depth in terms of meaning of the
            #          algorithm, but that is not what seems to be in the paper alg.
            #          -1 can be subtracted if we defined the distance between
            #          consecutive accesses (e.g. r, r) as 0 rather than 1.
            sd = l - i  # - 1
            # push r to the end of stack_distances
            stack_distances.appendleft(sd)
            # remove r from its position and insert to the top of stack
            del rstack[i]  # rstack.remove(r)
            rstack.append(r)
        except ValueError:  # not found #
            sd = 0  # -1
            # push r to the end of stack_distances/line_accesses
            stack_distances.appendleft(sd)
            line_accesses.appendleft(r)
            # push r to the top of stack
            rstack.append(r)

    if enable_padding:
        # WARNING: notice that as the ratio between the number of samples (l)
        # and cardinality [c] of a sample increases the probability of
        # generating a sample gets smaller and smaller because there are
        # few new samples compared to repeated samples. This means that for a
        # long trace with relatively small cardinality it will take longer to
        # generate all new samples and therefore obtain full distribution support
        # and hence it takes longer for distribution to resemble the original.
        # Therefore, we may pad the number of new samples to be on par with
        # average number of samples l/c artificially.
        l = len(stack_distances)
        c = max(stack_distances)
        padding = int(np.ceil(l / c))
        stack_distances = stack_distances + [0] * padding

    return (rstack, stack_distances, line_accesses)


# auxiliary read/write routines
def read_trace_from_file(file_path):
    try:
        with open(file_path) as f:
            if args.trace_file_binary_type:
                array = np.fromfile(f, dtype=np.uint64)
                trace = array.astype(np.uint64).tolist()
            else:
                line = f.readline()
                trace = list(map(lambda x: np.uint64(x), line.split(", ")))
            return trace
    except Exception:
        print(f"ERROR: trace file '{file_path}' is not available.")


def write_trace_to_file(file_path, trace):
    try:
        if args.trace_file_binary_type:
            with open(file_path, "wb+") as f:
                np.array(trace).astype(np.uint64).tofile(f)
        else:
            with open(file_path, "w+") as f:
                s = str(list(trace))
                f.write(s[1: len(s) - 1])
    except Exception:
        print("ERROR: no output trace file has been provided")


def read_dist_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
    except Exception:
        print("{file_path} Wrong file or file path")
    # read unique accesses
    unique_accesses = [int(el) for el in lines[0].split(", ")]
    # read cumulative distribution (elements are passed as two separate lists)
    list_sd = [int(el) for el in lines[1].split(", ")]
    cumm_sd = [float(el) for el in lines[2].split(", ")]

    return unique_accesses, list_sd, cumm_sd


def write_dist_to_file(file_path, unique_accesses, list_sd, cumm_sd):
    try:
        with open(file_path, "w") as f:
            # unique_acesses
            s = str(list(unique_accesses))
            f.write(s[1: len(s) - 1] + "\n")
            # list_sd
            s = str(list_sd)
            f.write(s[1: len(s) - 1] + "\n")
            # cumm_sd
            s = str(list(cumm_sd))
            f.write(s[1: len(s) - 1] + "\n")
    except Exception:
        print("Wrong file or file path")


if __name__ == "__main__":
    import argparse
    import operator

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Generate Synthetic Distributions")
    parser.add_argument("--trace-file", type=str, default="./input/trace.log")
    parser.add_argument("--trace-file-binary-type", type=bool, default=False)
    parser.add_argument("--trace-enable-padding", type=bool, default=False)
    parser.add_argument("--dist-file", type=str, default="./input/dist.log")
    parser.add_argument(
        "--synthetic-file", type=str, default="./input/trace_synthetic.log"
    )
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--print-precision", type=int, default=5)
    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    ### read trace ###
    trace = read_trace_from_file(args.trace_file)
    # print(trace)

    ### profile trace ###
    (_, stack_distances, line_accesses) = trace_profile(
        trace, args.trace_enable_padding
    )
    stack_distances.reverse()
    line_accesses.reverse()
    # print(line_accesses)
    # print(stack_distances)

    ### compute probability distribution ###
    # count items
    l = len(stack_distances)
    dc = sorted(
        collections.Counter(stack_distances).items(), key=operator.itemgetter(0)
    )

    # create a distribution
    list_sd = list(map(lambda tuple_x_k: tuple_x_k[0], dc))  # x = tuple_x_k[0]
    dist_sd = list(
        map(lambda tuple_x_k: tuple_x_k[1] / float(l), dc)
    )  # k = tuple_x_k[1]
    cumm_sd = deque()  # np.cumsum(dc).tolist() #prefixsum
    for i, (_, k) in enumerate(dc):
        if i == 0:
            cumm_sd.append(k / float(l))
        else:
            # add the 2nd element of the i-th tuple in the dist_sd list
            cumm_sd.append(cumm_sd[i - 1] + (k / float(l)))

    ### write stack_distance and line_accesses to a file ###
    write_dist_to_file(args.dist_file, line_accesses, list_sd, cumm_sd)

    ### generate corresponding synthetic ###
    # line_accesses, list_sd, cumm_sd = read_dist_from_file(args.dist_file)
    synthetic_trace = trace_generate_lru(
        line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    )
    # synthetic_trace = trace_generate_rand(
    #     line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    # )
    write_trace_to_file(args.synthetic_file, synthetic_trace)
