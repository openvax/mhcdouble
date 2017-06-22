from mhc2 import ModelCollection, Ensemble
from nose.tools import eq_

def test_new_model_collection_is_empty():
    mc = ModelCollection("_test")
    eq_(mc.alleles(), [])
    mc.delete()

def test_add_empty_ensemble():
    mc = ModelCollection("_test")
    e = Ensemble(models=[])
    mc.add_ensemble("DRB1*01:01", e)
    e2 = mc["DRB1*01:01"]
    eq_(e, e2)
    mc.delete()

