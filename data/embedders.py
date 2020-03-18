from gensim.models import KeyedVectors
import numpy as np
from defs import TRAIN_SET_PATH, GLOVE_TWITTER_25_PATH, VOCABULARY_FILE_PATH
import gensim.downloader as api
from scipy.sparse import csr_matrix


class GloveEmbedder:

    def __init__(self, embedding_size=25):
        super(GloveEmbedder, self).__init__()
        if embedding_size == 25:
            model = KeyedVectors.load_word2vec_format(GLOVE_TWITTER_25_PATH)
        else:
            raise NotImplementedError(f"Embedding size {embedding_size} not implemented yet")
        self.model = model
        self.non_existing_token = np.zeros(shape=(embedding_size,))

    def embed(self, tweet_tokens):
        vecs = []
        for token in tweet_tokens:
            try:
                vec = self.model.get_vector(token)
            except KeyError:
                vec = self.non_existing_token
            vecs.append(vec)
        return vecs


class OneHotEncoder:

    def __init__(self):
        super(OneHotEncoder, self).__init__()
        self.vocabulary = np.loadtxt(
            fname=VOCABULARY_FILE_PATH,
            dtype=str
        )
        self.voc_lookup = {k: v for v, k in enumerate(self.vocabulary) }

    def embed(self, twitter_tokens):
        one_hot = np.zeros(shape=len(self.vocabulary), dtype=np.float32)
        non_zero_indices = list(self.voc_lookup[token] for token in twitter_tokens if token in self.voc_lookup)
        np.put(one_hot, non_zero_indices, np.ones(shape=(len(non_zero_indices)), dtype=np.float32))
        return one_hot


def _save_twitter_glove_model():
    model = api.load("glove-twitter-25")
    model.save_word2vec_format(GLOVE_TWITTER_25_PATH)
