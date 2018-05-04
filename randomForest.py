import decision_tree
from classifier import classifier


class random_forest(classifier):
    def __init__(self, trees=10, max_depth=-1):
        super().__init__()
        self.trees = trees
        self.max_depth = max_depth

    def fit(self, X, y):
        # print("X=", X)
        # print("y=", y)
        pass

    def predict(self, X):
        pass
