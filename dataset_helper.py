from __future__ import absolute_import, division, print_function, unicode_literals

# others
from os import path
import sys
import bisect
import collections

# numpy
import numpy as np
from numpy import random as ra
from collections import deque


# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler
from typing import Tuple


def get_split_data(pro_data: str = "") -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    with np.load(pro_data) as data:
        X_int = data["X_int"]  # continuous  feature
        X_cat = data["X_cat"]  # categorical feature
        y = data["y"]          # target
        counts = data["counts"]

        print("X_int shape:", X_int.shape)
        print("X_cat shape:", X_cat.shape)
        print("y shape:", y.shape)
        print("counts shape:", counts.shape)
        print("counts:", counts)

        return X_int, X_cat, y, counts


if __name__ == "__main__":
    pro_data = "./input/kaggleAdDisplayChallenge_processed.npz"
    X_int, X_cat, y, counts = get_split_data(pro_data)
    print("X_int shape:", X_int.shape)
    print("X_cat shape:", X_cat.shape)
    print("y shape:", y.shape)
    print("counts shape:", counts.shape)
    print("counts:", counts)
