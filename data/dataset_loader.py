import numpy as np
import pandas as pd
from defs import TRAIN_SET_PATH, TEST_SET_PATH, DEV_SET_PATH


class DatasetLoader:

    @staticmethod
    def load_training_set():
        return DatasetLoader._load_dataset(TRAIN_SET_PATH)

    @staticmethod
    def load_testing_set():
        return DatasetLoader._load_dataset(TEST_SET_PATH)

    @staticmethod
    def load_development_set():
        return DatasetLoader._load_dataset(DEV_SET_PATH)

    @staticmethod
    def _load_dataset(data_path):
        ds = pd.read_csv(
            filepath_or_buffer=data_path,
            delimiter='\t',
        ).values
        tweets = ds[:, 1]
        emotions = ds[:, 2:].astype(dtype=float)
        return tweets, emotions
