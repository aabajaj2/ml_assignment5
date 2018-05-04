from decision_tree import *
from classifier import classifier
import numpy as np


class random_forest(classifier):
    def __init__(self, trees=10, max_depth=-1):
        super().__init__()
        self.trees = trees
        self.max_depth = max_depth

    def sample_of_features(self, X):
        random_features_indices = np.random.choice(len(X[0]), size=3, replace=False)
        return random_features_indices

    def subsample(self, X, y):
        subsample_x = []
        subsample_y = []
        return subsample_x, subsample_y

    def fit(self, X, y):
        tree_list = []
        for i in range(0, self.trees):
            tree_list.append(decision_tree())
        for dt in tree_list:
            subsample_x, subsample_y = self.subsample(X, y)  # Bagging
            feature_list = self.sample_of_features(X)   # Random features
            dt.fit(subsample_x, subsample_y, feature_list)  # Fit decision trees
            print(feature_list)
        pass

    def predict(self, X):
        pass
