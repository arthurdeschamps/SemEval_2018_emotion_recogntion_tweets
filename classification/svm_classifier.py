from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier
from classification.base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):

    def __init__(self, *args, **kwargs):
        super(SVMClassifier, self).__init__(*args, **kwargs)

    def get_classifier(self):
        # Best parameters according to grid search below
        return MultiOutputClassifier(
            estimator=svm.SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', verbose=True)
        )

    def get_classifier_for_grid_search(self):
        return MultiOutputClassifier(svm.SVC())

    def get_grid_search_parameters(self):
        parent_params_id = 'estimator__'
        return {
                f'{parent_params_id}C': [0.1, 1.0, 10.0],
                f'{parent_params_id}kernel': ['rbf', 'sigmoid'],
                f'{parent_params_id}class_weight': ['balanced', None],
                f'{parent_params_id}gamma': ['scale', 'auto'],
        }


#SVMClassifier(debug_mode=False).fit()
SVMClassifier().perform_grid_search()
