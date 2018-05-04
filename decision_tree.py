from classifier import classifier


class decision_tree(classifier):

    def __init__(self, criterion="entropy"):
        super().__init__()
        self.criterion = criterion

    def gini(self, Y):
        size = len(Y)
        counts = dict()
        for y in Y:
            if y not in counts:
                counts[y] = 0.
            counts[y] += 1.
        gini = 0.
        for key in counts:
            prob = counts[key] / size
            gini += prob * (1 - prob)
        return gini

    def entropy(self, Y):
        from math import log

        size = len(Y)
        counts = dict()
        for y in Y:
            if y not in counts:
                counts[y] = 0.
            counts[y] += 1.
        entropy = 0.
        for key in counts:
            prob = counts[key] / size
            entropy -= prob * log(prob, 2)
        return entropy

    def split_data(self, X, Y, axis, value):
        return_x = []
        return_y = []

        for x, y in (zip(X, Y)):
            if x[axis] == value:
                reduced_x = x[:axis]
                reduced_x.extend(x[axis + 1:])
                return_x.append(reduced_x)
                return_y.append(y)
        return return_x, return_y

    def choose_feature(self, X, Y):
        if self.criterion == "entropy":
            print("entropy")
            entropy = self.entropy(Y)
            best_information_gain = 0.
            best_feature = -1
            for i in range(len(X[0])):  # For each feature
                feature_list = [x[i] for x in X]
                values = set(feature_list)
                entropy_i = 0.
                for value in values:
                    sub_x, sub_y = self.split_data(X, Y, i, value)
                    prob = len(sub_x) / float(len(X))
                    entropy_i += prob * self.entropy(sub_y)
                info_gain = entropy - entropy_i
                if info_gain > best_information_gain:
                    best_information_gain = info_gain
                    best_feature = i
            return best_feature
        else:
            print("gini")
            gini = self.gini(Y)
            best_information_gain = 0.
            best_feature = -1
            for i in range(len(X[0])):  # For each feature
                feature_list = [x[i] for x in X]
                values = set(feature_list)
                gini_i = 0.
                for value in values:
                    sub_x, sub_y = self.split_data(X, Y, i, value)
                    prob = len(sub_x) / float(len(X))
                    gini_i += prob * self.gini(sub_y)
                info_gain = gini - gini_i
                if info_gain > best_information_gain:
                    best_information_gain = info_gain
                    best_feature = i
            return best_feature


    def class_dict(self, Y):
        classes = dict()
        for y in Y:
            if y not in classes:
                classes[y] = 0
            classes[y] += 1
        return classes


    def majority(self, Y):
        from operator import itemgetter
        # Use this function if a leaf cannot be split further and
        # ... the node is not pure

        classcount = self.class_dict(Y)
        sorted_classcount = sorted(classcount.items(), key=itemgetter(1), reverse=True)
        return sorted_classcount[0][0]

    def build_tree(self, X, Y):
        # IF there's only one instance or one class, don't continue to split
        if len(Y) <= 1 or len(self.class_dict(Y)) == 1:
            return self.majority(Y)

        if len(X[0]) == 1:
            return self.majority(Y) 

        best_feature = self.choose_feature(X, Y)
        if best_feature < 0 or best_feature >= len(X[0]):
            return self.majority(Y)

        this_tree = dict()
        feature_values = [example[best_feature] for example in X]
        unique_values = set(feature_values)
        for value in unique_values:
            # Build a node with each unique value:
            subtree_x, subtree_y = self.split_data(X, Y, best_feature, value)
            if best_feature not in this_tree:
                this_tree[best_feature] = dict()
            if value not in this_tree[best_feature]:
                this_tree[best_feature][value] = 0
            this_tree[best_feature][value] = self.build_tree(subtree_x, subtree_y)
        return this_tree

    def predict_value(self, x_data, tree):
        if isinstance(tree, int):
            return tree
        indices = list(tree.keys())

        for index in indices:
            keys = list(tree[index].keys())
            for key in keys:
                if x_data[index] == key:
                    sub_tree = tree.get(index).get(key)
                    return self.predict_value(x_data, sub_tree)

    def fit(self, X, Y):
        self.tree = self.build_tree(X, Y)

    def predict(self, X):
        hyp = []
        for x in X:
            pre = self.predict_value(x, self.tree)
            hyp.append(pre)
        return hyp