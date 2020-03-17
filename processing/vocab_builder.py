import numpy as np
from defs import TRAIN_SET_PATH, DEV_SET_PATH
from processing.processing_pipeline import ProcessingPipeline


def extract_vocabulary(data_path=TRAIN_SET_PATH):
    df = np.loadtxt(
        fname=data_path,
        dtype=str,
        delimiter='	',
        usecols=[1],
        skiprows=1
    )
    pipeline = ProcessingPipeline.standard_pipeline(df)
    pipeline.process()
    pipeline.build_vocabulary()
    pipeline.print_processed_dataset()
    print(pipeline.vocabulary)


extract_vocabulary(DEV_SET_PATH)
