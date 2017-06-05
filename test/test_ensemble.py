from mhc2 import Ensemble
from nose.tools import eq_
import numpy as np

class DummyPredictor(object):
    def __init__(self, const_pred_value=True):
        self.const_pred_value = const_pred_value

    def predict(self, peptides):
        n = len(peptides)
        return np.array([self.const_pred_value] * n)


def test_ensemble_model_weights():
    e = Ensemble(models=[None])
    eq_(len(e.model_weights), 1)
    eq_(e.model_weights[0], 1)

def test_ensemble_predict():
    p_true = DummyPredictor(True)
    p_false = DummyPredictor(False)
    e = Ensemble([p_true, p_false])
    pred = e.predict(["SIINFEKL"])
    eq_(len(pred), 1)
    eq_(pred[0], 0.5)
