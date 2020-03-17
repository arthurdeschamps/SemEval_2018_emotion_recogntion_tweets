import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EC_DATA_PATH = f"{ROOT_DIR}/SemEval2018-Task1-all-data/English/E-c"
TRAIN_SET_PATH = f"{EC_DATA_PATH}/2018-E-c-En-train.txt"
TEST_SET_PATH = f"{EC_DATA_PATH}/2018-E-c-En-test-gold.txt"
DEV_SET_PATH = f"{EC_DATA_PATH}/2018-E-c-En-dev.txt"
PROCESSED_DATA_PATH = f"{ROOT_DIR}/processed_data"

STOP_WORDS_FILE_PATH = f"{ROOT_DIR}/processing/nltk_stopwords.txt"
