from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score, f1_score, classification_report
from data.dataset_loader import DatasetLoader
from processing.processing_pipeline import ProcessingPipeline
from data.stats import show_labels_stats


class BaseClassifier:

    def __init__(self, *args, debug_mode=False, **kwargs):
        super(BaseClassifier, self).__init__(*args, **kwargs)

        if debug_mode:
            training_set = DatasetLoader.load_development_set()
        else:
            training_set = DatasetLoader.load_training_set()

        self.train_embeddings, self.train_labels = BaseClassifier._prepare_data(training_set)
        self.test_embeddings, self.test_labels = BaseClassifier._prepare_data(DatasetLoader.load_testing_set())

        self.scaler = StandardScaler()
        self.train_embeddings = self.scaler.fit_transform(self.train_embeddings)
        self.test_embeddings = self.scaler.transform(self.test_embeddings)

    def report_performance(self, predictions):
        print(f"Accuracy: {self.jaccard_index(predictions)}")
        print(f"F1-score micro: {self.micro_f1(predictions)}")
        print(f"F1-score macro: {self.macro_f1(predictions)}")
        # print(classification_report(self.test_labels, predictions))

    def jaccard_index(self, y_pred):
        return jaccard_score(self.test_labels, y_pred, average='micro')

    def micro_f1(self, y_pred):
        return f1_score(self.test_labels, y_pred, average='micro')

    def macro_f1(self, y_pred):
        return f1_score(self.test_labels, y_pred, average='macro')

    @staticmethod
    def report_prediction_stats(predictions):
        show_labels_stats(predictions)

    @staticmethod
    def _prepare_data(dataset):
        tweets, emotions = dataset
        data_pipeline = ProcessingPipeline.standard_pipeline(tweets)
        data_pipeline.process()
        data_pipeline.embed()
        return data_pipeline.embeddings, emotions
