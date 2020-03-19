from typing import List

from defs import TRAIN_SET_PATH, VOCABULARY_FREQUENCIES_FILE_PATH, VOCABULARY_FILE_PATH
from processing.processing_pipeline import ProcessingPipeline
from data.dataset_loader import DatasetLoader


def extract_vocabulary_frequencies(data_path=TRAIN_SET_PATH):
    with open(VOCABULARY_FREQUENCIES_FILE_PATH, "w") as f:
        voc = _extract_vocabulary(data_path)
        for word, occ in voc:
            f.write(f"{word} {occ}\n")


def create_term_dict(dataset: List[List[str]] = None, min_occurrences=4):
    """
    :dataset (Optional) Dataset to extract the vocabulary from. If None, will use SemEval2018 english
    training set as default.
    :min_occurrences Minimum number of occurrences for a word to be saved in the vocabulary.
    """
    with open(VOCABULARY_FILE_PATH, "w") as f:
        voc = _extract_vocabulary(DatasetLoader.load_training_set()[0] if dataset is None else dataset)
        for word, occ in voc:
            if occ > min_occurrences:
                f.write(f"{word}\n")


def _extract_vocabulary(df):
    pipeline = ProcessingPipeline.standard_pipeline(df)
    pipeline.process()
    pipeline.build_vocabulary()
    return list((k, v) for k, v in sorted(pipeline.vocabulary.items(), key=lambda item: -item[1]))


create_term_dict()
# extract_vocabulary_frequencies()
