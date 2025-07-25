import numpy as np

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "pvenieris3"

    def study_group(self):
        return "pvenieris3"

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def query(self, points):
        def traverse_tree(point, node_index=0):
            node = self.tree[node_index]
            if node[0] == -1:
                return node[1]
            feature = int(node[0])
            number = node[1]
            if point[feature] <= number:
                return traverse_tree(point, node_index + 1)
            else:
                return traverse_tree(point, node_index + int(node[3]))
        final = [traverse_tree(point) for point in points]
        return final

    def build_tree(self,data_x, data_y):
        if data_x.shape[0] <= self.leaf_size or len(np.unique(data_y)) == 1:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        std_x = np.std(data_x, axis=0)
        std_y = np.std(data_y)
        mask = std_x != 0
        correlations = np.zeros(data_x.shape[1])
        if std_y != 0:
            correlations[mask] = np.abs(np.corrcoef(data_x[:, mask], data_y, rowvar=False)[0:mask.sum(), -1])
        feature = np.argmax(correlations)
        number = np.median(data_x[:, feature])
        if np.all(data_x[:, feature] == data_x[0, feature]):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        left = data_x[:, feature] <= number
        right = data_x[:, feature] > number
        if np.sum(left) == 0 or np.sum(right) == 0:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        left_tree = self.build_tree(data_x[left], data_y[left])
        right_tree = self.build_tree(data_x[right], data_y[right])
        root = np.array([[feature, number, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))