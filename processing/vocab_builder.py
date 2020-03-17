import numpy as np
from defs import TRAIN_SET_PATH, DEV_SET_PATH


def extract_vocabulary(data_path=TRAIN_SET_PATH):
    df = np.loadtxt(
        fname=data_path,
        dtype=str,
        delimiter='	',
        usecols=[1],
        skiprows=1
    )
    print(df)


extract_vocabulary(DEV_SET_PATH)