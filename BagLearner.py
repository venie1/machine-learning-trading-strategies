import numpy as np
import pandas as pd
import scipy.stats as stats
class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []


        for _ in range(bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return "pvenieris3"  

    def study_group(self):
        return "pvenieris3"  

    def add_evidence(self, data_x, data_y):
        n = data_x.shape[0]
        if n != data_y.shape[0]:
            raise ValueError(f"Mismatch in lengths: data_x has {n} rows, data_y has {data_y.shape[0]} rows.")

        for i in range(self.bags):
            indices = np.random.choice(n, size=n, replace=True)
            if isinstance(data_x, pd.DataFrame):
                sample_x = data_x.iloc[indices]
            else:
                sample_x = data_x[indices]
            sample_y = data_y[indices]
            self.learners[i].add_evidence(sample_x, sample_y)

    def query(self, points):
        predictions = np.zeros((self.bags, points.shape[0]))
        for i in range(self.bags):
            predictions[i] = self.learners[i].query(points)

        if not self.boost:
            return stats.mode(predictions, axis=0)[0][0]  # Majority voting
        return np.mean(predictions, axis=0)  # For boosting, we return the mean (used for regression)
