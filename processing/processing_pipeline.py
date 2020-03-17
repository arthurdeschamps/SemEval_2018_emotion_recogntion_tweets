import processing.operators as ops


class ProcessingPipeline:

    def __init__(self, dataset, *operators):
        super(ProcessingPipeline, self).__init__()
        self.dataset = dataset
        self.operators = list(operators)
        self.processed_tweets = None
        self.vocabulary = None

    def process(self):
        processed = []
        for tweet in self.dataset:
            processed.append(self.process_tweet(tweet))
        self.processed_tweets = processed

    def process_tweet(self, tweet):
        for operator in self.operators:
            tweet = operator(tweet)
        return tweet

    def build_vocabulary(self):
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
        standard_operators = (
            ops.demojize,
            ops.tokenize,
            ops.pos_tag,
            ops.lemmatize,
            ops.name_entity_recognition,
            ops.remove_stop_words,
            ops.substitute_numbers,
            ops.to_lowercase,
            ops.substitute_punctuation,
            ops.substitute_tweeter_usernames,
        )
        return ProcessingPipeline(dataset, *standard_operators)

