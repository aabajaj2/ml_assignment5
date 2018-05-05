from decision_tree import *
from classifier import classifier
import numpy as np
from collections import defaultdict
import operator

class random_forest(classifier):
    def __init__(self, trees=10, max_depth=-1):
        super().__init__()
        self.trees = trees
        self.max_depth = max_depth
        self.tree_list = None

    def sample_of_features(self, X):
        random_features_indices = np.random.choice(len(X[0]), size=3, replace=False)
        return random_features_indices

    def subsample(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        subsample_x = []
        subsample_y = []
        for index in indices:
            subsample_x.append(X[index])
            subsample_y.append(y[index])
        return subsample_x, subsample_y

    def fit(self, X, y):
        tree_list = []
        for i in range(0, self.trees):
            tree_list.append(decision_tree())
        for dt in tree_list:
            subsample_x, subsample_y = self.subsample(X, y)  # Bagging
            dt.feature_list = self.sample_of_features(X)   # Random features
            dt.fit(subsample_x, subsample_y)  # Fit decision trees
            # print(dt.feature_list)
        self.tree_list = tree_list

    def predict(self, X):
        print("Tree list", self.tree_list)
        hypothesis_list = [t.predict(X) for t in self.tree_list]
        print("hyp lst=", hypothesis_list)
        hyp = []
        for i in range(0, len(X)):
            counts = defaultdict(int)
            for h in hypothesis_list:
                counts[h[i]] += 1
                # print("H[i]", h[i])
            get_max_value = max(counts.items(), key=operator.itemgetter(1))[0]
            hyp.append(get_max_value)
        return hyp