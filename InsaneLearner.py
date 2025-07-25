import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.bag_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=verbose)
    def add_evidence(self, Xtrain, Ytrain):
        self.bag_learner.add_evidence(Xtrain, Ytrain)
    def query(self, Xtest):
        return self.bag_learner.query(Xtest)
    def author(self):
        return "pvenieris3"
    def study_group(self):
        return "pvenieris3"