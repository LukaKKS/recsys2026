# LEAF: Lightweight, Efficient, Adaptive and Flexible Embedding for Large-Scale Recommendation Models

This is the source code for the paper LEAF: Lightweight, Efficient, Adaptive and Flexible Embedding for Large-Scale Recommendation Models (RecSys 2025). In this work, we propose a multi-level hashing framework that compresses the large embedding tables based on access frequency.

## Usage and Examples

Our implementation builds upon DLRM repo: https://github.com/facebookresearch/dlrm, and CAFE repo: https://github.com/HugoZHL/CAFE. Using the scripts will allow you to run LEAF and all the included baselines, namely full embedding without any compression, Hashing Trick and CAFE for all datasets. The repository also supports WDL and DCN.

1. The code supports interface with the [Criteo Kaggle Display Advertising Challenge Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).

   - The model can be trained using the following script

     - Follow the instruction on Facebook DLRM to generate kaggleAdDisplayChallenge_processed.npz
     - Set the parameters --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz in the script.

     ```
     ./bench/criteo_kaggle.sh
     ```

2. The code supports interface with the [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).

   - The model can be trained using the following script

     - Follow the instruction on Facebook DLRM to generate terabyte_processed.npz
     - Set the parameters --processed-data-file=./input/terabyte_processed.npz in the script.

     ```
     ./bench/criteo_terabyte.sh
     ```

3. The code also supports another two datasets [Avazu](https://kaggle.com/competitions/avazu-ctr-prediction) and [KDD12](https://kaggle.com/competitions/kddcup2012-track2).
   - Please do the following to prepare the dataset for use with this code:
     - Set the parameters cat_path, dense_path, label_path and count_path in the script.

   - The model can be trained using the following script

     ```
     ./bench/avazu.sh
     ./bench/kdd12.sh
     ```

4. The code provides three models to train the dataset:
   - dlrm:

     ```
     ./bench/criteo_kaggle.sh
     ```
   - wdl:

     ```
     ./bench/wdl.sh
     ```
   - dcn:

     ```
     ./bench/dcn.sh
     ```

4. The code provides methods for generating baseline embedding layers:

   - Full embedding with the following script

     ```
     ./bench/criteo_kaggle.sh
     ```

   - Hashing Trick with the following script

     ```
     ./bench/criteo_kaggle.sh "--hash-flag --compress-rate=0.001"
     ```

   - CAFE with the following script

     ```
     ./bench/criteo_kaggle.sh "--sketch-flag --compress-rate=0.001 --hash-rate=0.3"
     ```
    
   - Q-R Trick with the following script

     ```
     ./bench/criteo_kaggle.sh "--qr-flag --qr-collisions=10"
     ```


## LEAF Setup

Use scripts in the bench directory to train models with LEAF:
  ```
  ./bench/criteo_kaggle.sh "0" "--arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --use-adaptive-encoding --compression-ratio=1000 --long-tail-memory-ratio=0.9" "" > output.log 2>&1 &
  ```

### Parameters
| name | explanation |
|----------|----------|
| --use-adaptive-encoding(bool) | use LEAF method for embedding compression|
| --compression-ratio(int) | ratio between memory usage of uncompressed embeddings and that of compressed embeddings|
| --long-tail-memory-ratio(float) | ratio between long-tail memory and the total memory|


## Contact
If you have any questions, feel free to contact me through email (chaoyij@usc.edu).
