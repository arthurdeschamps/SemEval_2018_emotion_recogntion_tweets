from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier
from processing.processing_pipeline import ProcessingPipeline
from data.dataset_loader import DatasetLoader


class SVMClassifier:

    def __init__(self):
        super(SVMClassifier, self).__init__()

        def _prepare_data(dataset):
            tweets, emotions = dataset
            data_pipeline = ProcessingPipeline.standard_pipeline(tweets)
            data_pipeline.process()
            data_pipeline.embed()
            return data_pipeline.embeddings, emotions
        self.train_embeddings, self.train_labels = _prepare_data(DatasetLoader.load_development_set())
        self.test_embeddings, self.test_labels = _prepare_data(DatasetLoader.load_testing_set())

    def fit(self):
        estimator = svm.SVC(gamma=1e-8, class_weight='balanced')
        classifier = MultiOutputClassifier(estimator=estimator)
        classifier.fit(self.train_embeddings, self.train_labels)
        preds = classifier.predict(self.test_embeddings)
        for pred in preds:
            print(sum(pred))
        exit()


SVMClassifier().fit()
