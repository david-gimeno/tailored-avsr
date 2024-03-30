import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    """
      Incremental computation of the mean and standard deviation of a dataset. Code mainly based on:
    https://dsp.stackexchange.com/questions/811/determining-the-mean-and-standard-deviation-in-real-time
    """

    parser = argparse.ArgumentParser(description="Incremental computation of the mean and standard deviation of a ROI dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--training-dataset", required=True, type=str, help="Path to where the training dataset split is")
    args = parser.parse_args()

    lips_paths = pd.read_csv(args.training_dataset)["lips_path"].tolist()

    n = 0
    psum = 0.0
    psum_sq = 0.0
    for lips_path in tqdm(lips_paths):
        # -- loading and normalizing
        lips = np.load(lips_path)["data"]
        lips = lips / 250.0

        # -- incremental computation of mean and standard deviation
        if lips.shape[0] != 0:
            n += lips.shape[0]
            psum += lips.sum()
            psum_sq += (lips**2).sum()

    count = n * 88 * 88
    mean = psum / count
    var = (psum_sq / count) - (mean**2)
    std = var**0.5

    print(f"MEAN: {mean} || VAR: {var} || STD: {std}")
