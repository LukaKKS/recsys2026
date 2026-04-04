import argparse
import sys
import numpy as np
import dataset_helper


def analyze_dataset(args):
    if args.data_set == 'kdd12':
        data_cat = np.memmap(args.cat_path, dtype=np.int32,
                             mode='r', shape=(149639105, 11))
        data_T = np.memmap(args.label_path, dtype=np.int32,
                           mode='r', shape=(149639105,))
        data_int = None
        data_count = np.memmap(
            args.count_path, dtype=np.int32, mode='r', shape=(12,))

        count = np.array(data_count)
        new_count = np.zeros(11)
        for i in range(11):
            new_count[i] = count[i+1] - count[i]
        # print(f"new_count: {new_count}")
        np.random.seed(2023)
        tot_len = 149639105
        index = np.arange(tot_len)
        np.random.shuffle(index)
        # print(f"index: {index.shape} {index}")

        sum_count = np.zeros(11, dtype=np.int32)
        for i in range(1, 11):
            sum_count[i] = new_count[i-1] + sum_count[i-1]

        result_cat = data_cat - sum_count
        i = args.table_index
        data_column = result_cat[:, i]
        values, counts = np.unique(data_column, return_counts=True)

        sorted_indices = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_indices]
        n = len(sorted_counts)
        end = n if n <= 5 else n // 5
        top_twenty_sum_counts = sum(sorted_counts[:end])
        print(f"table {i} top_twenty_sum_counts: {top_twenty_sum_counts} tot_len: {tot_len} ratio: {top_twenty_sum_counts / tot_len}")
        
    elif args.data_set == 'kaggle':
        pro_data = "./input/kaggleAdDisplayChallenge_processed.npz"
        data_int, data_cat, data_T, data_count = dataset_helper.get_split_data(pro_data)
        i = args.table_index
        data_column = data_cat[:, i]
        values, counts = np.unique(data_column, return_counts=True)
        tot_len = len(data_T)

        sorted_indices = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_indices]
        n = len(sorted_counts)
        end = n if n <= 5 else n // 5
        top_twenty_sum_counts = sum(sorted_counts[:end])
        print(f"table {i} top_twenty_sum_counts: {top_twenty_sum_counts} tot_len: {tot_len} ratio: {top_twenty_sum_counts / tot_len}")


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    parser.add_argument("--data-set", type=str,
                        default="kaggle")  # or terabyte
    parser.add_argument("--cat-path", type=str,
                        default="../criteo_24days/sparse")
    parser.add_argument("--dense-path", type=str,
                        default="../criteo_24days/dense")
    parser.add_argument("--label-path", type=str,
                        default="../criteo_24days/label")
    parser.add_argument("--count-path", type=str,
                        default="../criteo_24days/processed_count.bin")
    parser.add_argument("--table-index", type=int, default=0)

    args = parser.parse_args()
    analyze_dataset(args)


if __name__ == "__main__":
    run()