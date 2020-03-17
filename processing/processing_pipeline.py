import processing.operators as ops

class ProcessingPipeline:

    def __init__(self, dataset, *operators):
        super(ProcessingPipeline, self).__init__()
        self.dataset = dataset
        self.operators = list(operators)

    def process(self):
        pass  # TODO: process each tweet, save processed dataset

    def process_tweet(self, tweet):
        for operator in self.operators:
            tweet = operator(tweet)
        return tweet

    @staticmethod
    def process_standard(dataset):
        standard_operators = (
            ops.demojize,
            ops.to_lowercase,
            ops.erase_numbers,
            ops.erase_punctuation,
            ops.tokenize,
            ops.remove_stop_words,
            ops.pos_tag,
            ops.lemmatize,
        )
        return ProcessingPipeline(dataset, *standard_operators)
