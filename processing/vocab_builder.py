from defs import TRAIN_SET_PATH, VOCABULARY_FREQUENCIES_FILE_PATH, VOCABULARY_FILE_PATH
from processing.processing_pipeline import ProcessingPipeline
from data.dataset_loader import DatasetLoader


def extract_vocabulary_frequencies(data_path=TRAIN_SET_PATH):
    with open(VOCABULARY_FREQUENCIES_FILE_PATH, "w") as f:
        voc = _extract_vocabulary(data_path)
        for word, occ in voc:
            f.write(f"{word} {occ}\n")


def extract_vocabulary(data_path=TRAIN_SET_PATH):
    with open(VOCABULARY_FILE_PATH, "w") as f:
        voc = _extract_vocabulary(data_path)
        for word, occ in voc:
            if occ > 4:
                f.write(f"{word}\n")


def _extract_vocabulary():
    df = DatasetLoader.load_training_set()
    pipeline = ProcessingPipeline.standard_pipeline(df)
    pipeline.process()
    pipeline.build_vocabulary()
    return list((k, v) for k, v in sorted(pipeline.vocabulary.items(), key=lambda item: -item[1]))


# extract_vocabulary(TRAIN_SET_PATH)
# extract_vocabulary_frequencies()
