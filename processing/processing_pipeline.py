import processing.operators as ops
from data.embedders import OneHotEncoder


class ProcessingPipeline:
    """
    This processing pipeline allows to transform a raw dataset of tweets into a usable, processed dataset.
    To get the feature vectors associated with the dataset you'll pass in argument, follow these steps:
    1. Initialize a ProcessingPipeline object (standard_pipeline can do it for you)
    2. Call "process()". This will transform the initial dataset using the defined pipeline. The processed data will be
    accessible through _.processed_tweets.
    3. Call "embed()". This will encode / create feature vectors from the processed tweets. They will then be available
    through _.embeddings.
    """

    def __init__(self, dataset, *operators, embedder=OneHotEncoder):
        super(ProcessingPipeline, self).__init__()
        self.dataset = dataset
        self.operators = list(operators)
        self.embedder = embedder()
        self.processed_tweets = None
        self.embeddings = None
        self.vocabulary = None

    def process(self):
        """
        Transform the initial dataset using the defined pipeline.
        """
        processed = []
        for tweet in self.dataset:
            processed.append(self.process_tweet(tweet))
        self.processed_tweets = processed

    def process_tweet(self, tweet):
        for operator in self.operators:
            tweet = operator(tweet)
        return tweet

    def embed(self):
        """
        Encodes / creates feature vectors from the processed tweets.
        """
        embeddings = []
        for tweet_tokens in self.processed_tweets:
            embeddings.append(self.embedder.get_feature_vector(tweet_tokens))
        self.embeddings = embeddings

    def build_vocabulary(self):
        """
        Builds a dictionary of words to occurrences from the processed tweets.
        """
        vocab = {}
        for processed_tweet in self.processed_tweets:
            for token in processed_tweet:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
        self.vocabulary = vocab

    def print_processed_dataset(self):
        for processed_tweet in self.processed_tweets:
            print(processed_tweet)

    @staticmethod
    def standard_pipeline(dataset):
        """
        Creates a usable ProcessingPipeline object using conventional data transformation operators.
        """
        standard_operators = (
            ops.demojize,
            ops.substitute_underscores,
            ops.tokenize,
            ops.pos_tagging,
            ops.lemmatize,
            ops.name_entity_recognition,
            ops.remove_stop_words,
            ops.substitute_numbers,
            ops.to_lowercase,
            ops.remove_punctuation,
            ops.strip_hashtags,
            ops.substitute_tweeter_usernames,
        )
        return ProcessingPipeline(dataset, *standard_operators)
