import decision_tree
import classifier


class randomForest(classifier):
    def __init__(self, trees=10, max_depth=-1):
        super().__init__()
        self.trees = trees
        self.max_depth = max_depth

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass
