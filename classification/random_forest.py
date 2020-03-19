from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from classification.base_classifier import BaseClassifier


class RandomForest(BaseClassifier):

    def __init__(self, *args, **kwargs):
        super(RandomForest, self).__init__(*args, **kwargs)

    def get_classifier(self):
        return MultiOutputClassifier(
            estimator=RandomForestClassifier(
                class_weight='balanced',
                n_estimators=100,
                criterion='gini',
                max_features='sqrt'
            )
        )

    def get_classifier_name(self) -> str:
        return "Random Forest Classifier"

    def get_classifier_for_grid_search(self):
        return MultiOutputClassifier(RandomForestClassifier())

    def get_grid_search_parameters(self):
        parent_params_id = 'estimator__'
        return {
            f'{parent_params_id}n_estimators': [10, 50, 100, 300],
            f'{parent_params_id}criterion': ['gini', 'entropy'],
            f'{parent_params_id}max_depth': [None, 10, 50],
            f'{parent_params_id}max_features': [None, 'sqrt'],
            f'{parent_params_id}verbose': [True],
            f'{parent_params_id}class_weight': [None, 'balanced']
        }


RandomForest(debug_mode=False, scale_data=False).fit()
