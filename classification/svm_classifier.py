from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from classification.base_classifier import BaseClassifier
from sklearn.model_selection import GridSearchCV


class SVMClassifier(BaseClassifier):

    def __init__(self, *args, **kwargs):
        super(SVMClassifier, self).__init__(*args, **kwargs)

    def fit(self):
        estimator = svm.SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', verbose=True)
        classifier = MultiOutputClassifier(estimator=estimator)
        classifier.fit(self.train_embeddings, self.train_labels)
        preds = classifier.predict(self.test_embeddings)
        self.report_performance(preds)
        self.report_prediction_stats(preds)

    def grid_search(self):
        parent_params_id = 'estimator__'
        clf = GridSearchCV(
            estimator=MultiOutputClassifier(svm.SVC()),
            param_grid={
                f'{parent_params_id}C': [1.0, 10.0],
                f'{parent_params_id}kernel': ['rbf', 'linear', 'sigmoid'],
                f'{parent_params_id}class_weight': ['balanced', None],
                f'{parent_params_id}gamma': ['scale', 'auto'],
            },
            cv=5
        )
        clf.fit(self.train_embeddings, self.train_labels)
        SVMClassifier._print_grid_search_results(clf, self.test_embeddings, self.test_labels)

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


SVMClassifier(debug_mode=False).fit()
#SVMClassifier().grid_search()
