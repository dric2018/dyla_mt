import argparse

from config import Config

import logging

import os
import os.path as osp

import numpy as np
import pandas as pd

from utils.utils import prepare_data

logging.basicConfig(level="INFO")

# argument parser
parser = argparse.ArgumentParser()

parser.add_argument('-s', "--save_data", default=False)    

if __name__ == "__main__":

    # argument parser
    args = parser.parse_args()
    save_data = args.save_data

    logging.info("Loading data")
    train_df    = pd.read_parquet(osp.join(Config.DATA_DIR, "train-00000-of-00001.parquet"))
    valid_df    = pd.read_parquet(osp.join(Config.DATA_DIR, "validation-00000-of-00001.parquet"))
    # test_df     = pd.read_parquet(osp.join(Config.DATA_DIR, "test-00000-of-00001.parquet"))

    logging.info("\n> Data visualization (before preprocessing)")
    print(train_df.sample(n=3, random_state=Config.SEED))

    logging.info(f"> # Train samples: {train_df.shape}, # Validation samples: {valid_df.shape}")

    # preprocess data
    train = prepare_data(df=train_df)
    valid = prepare_data(df=valid_df)
    # test = prepare_data(df=test_df)

    logging.info("\n> Data visualization (after preprocessing)")
    print(train.sample(n=3, random_state=Config.SEED))

    # save preprocessed data
    if save_data:
        logging.info("Saving preprocessed data")
        train.to_csv(osp.join(Config.DATA_DIR, "preprocessed/train.csv"), index=False)
        valid.to_csv(osp.join(Config.DATA_DIR, "preprocessed/valid.csv"), index=False)
        # test.to_csv(osp.join(DATA_DIR, "preprocessed/test.csv"), index=False)
        logging.info("Preprocessed data saved")
