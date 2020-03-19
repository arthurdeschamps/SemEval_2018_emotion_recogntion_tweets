from abc import ABC

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix
from data.dataset_loader import DatasetLoader
from processing.processing_pipeline import ProcessingPipeline
from data.stats import show_labels_stats


class BaseClassifier(ABC):
    """
    Class to extend to build new classifiers.
    """

    def __init__(self, *args, scale_data=False, debug_mode=False, **kwargs):
        """
        :scale_data If true, scales the data to mean 0 and variance 1. In practice, it seems that the performance are
        better without scaling.
        :debug_mode If true, uses the dev set instead of the train set for training this model.
        """
        super(BaseClassifier, self).__init__(*args, **kwargs)

        if debug_mode:
            training_set = DatasetLoader.load_development_set()
        else:
            training_set = DatasetLoader.load_training_set()

        self.train_embeddings, self.train_labels = BaseClassifier._prepare_data(training_set)
        self.test_embeddings, self.test_labels = BaseClassifier._prepare_data(DatasetLoader.load_testing_set())
        self.dev_embeddings, self.dev_labels = BaseClassifier._prepare_data(DatasetLoader.load_development_set())

        if scale_data:
            self.scaler = StandardScaler()
            self.train_embeddings = self.scaler.fit_transform(self.train_embeddings)
            self.test_embeddings = self.scaler.transform(self.test_embeddings)
            self.dev_embeddings = self.scaler.transform(self.dev_embeddings)

    def fit(self):
        """
        Call this method to train the classifier and report performances.
        """
        classifier = self.get_classifier()
        classifier.fit(self.train_embeddings, self.train_labels)
        preds = classifier.predict(self.test_embeddings)
        self.report_performance(preds)
        self.report_prediction_stats(preds)
        self.plot_confusion_matrix(preds)

    def perform_grid_search(self):
        clf = GridSearchCV(
            estimator=self.get_classifier_for_grid_search(),
            param_grid=self.get_grid_search_parameters(),
            cv=5
        )
        clf.fit(self.dev_embeddings, self.dev_labels)
        BaseClassifier._print_grid_search_results(clf, self.test_embeddings, self.test_labels)

    def get_classifier(self):
        """
        Must return a classifier which implements the fit(data, labels) and predict(data) methods.
        """
        raise NotImplementedError()

    def get_classifier_name(self) -> str:
        """
        Name of the classifier.
        """
        raise NotImplementedError()

    def get_classifier_for_grid_search(self):
        """
        Must return a classifier as for get_classifier but this one doesn't need any specific parameters, as they will
        be searched.
        """
        raise NotImplementedError()

    def get_grid_search_parameters(self):
        raise NotImplementedError()

    @staticmethod
    def _print_grid_search_results(clf, x_test, y_test):
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))
        print()

    def plot_confusion_matrix(self, predictions):
        binary_confusion_matrices = multilabel_confusion_matrix(self.test_labels, predictions)
        print(f"{self.get_classifier_name()} - Confusion Matrix")
        emotions = ["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Love", "Optimism", "Pessimism", "Sadness",
                    "Surprise", "Trust"]
        for i in range(len(emotions)):
            conf_mat = binary_confusion_matrices[i]
            print(f"{emotions[i]}")
            print("| Positive | Negative |")
            print("|:---------:|:-------:|")
            print(f"| {conf_mat[0][0]} | {conf_mat[0][1]} |")
            print(f"| {conf_mat[1][0]} | {conf_mat[1][1]} |")
            print()

    def report_performance(self, predictions):
        print(f"Accuracy: {self.jaccard_index(predictions)}")
        print(f"F1-score micro: {self.micro_f1(predictions)}")
        print(f"F1-score macro: {self.macro_f1(predictions)}")
        # print(classification_report(self.test_labels, predictions))

    def jaccard_index(self, predictions):
        assert len(self.test_labels) == len(predictions)
        jaccard_score = 0.0
        for pred_ind in range(len(predictions)):
            y_true = self.test_labels[pred_ind]
            y_pred = predictions[pred_ind]
            intersection_size = sum(1 if y_pred[i] == y_true[i] == 1 else 0 for i in range(len(y_pred)))
            union_size = sum(1 if (y_pred[i] == 1) or (y_true[i] == 1) else 0 for i in range(len(y_pred)))
            if intersection_size > 0:
                jaccard_score += intersection_size / (union_size * len(self.test_labels))
        return jaccard_score

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
