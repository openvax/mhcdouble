from mhc2.binding_core import BindingCorePredictor
from nose.tools import eq_

def test_binding_core_predictor():
    bcp = BindingCorePredictor()
    # constructing an example where the binding core is
    #   SIINFEKLF
    # and we get hits with this sequence and various extra
    # amino acids on the sides
    SIINFEKLF_hits = [
        "AFLIQQQSIINFEKLF",
        "SIINFEKLFLLL",
        "SIINFEKLFLLLYY",
        "QSIINFEKLF"
    ]
    # Making a second group of hits with a binding core that shares
    # Serine at the 1st residue,
    # Isoleucine at the 3rd residue
    # Leucine at the 8th residue
    #   SQIQQQQLQ
    SQIQQQQLQ_hits = [
        "SQIQQQQLQAYYY",
        "AAAAASQIQQQQLQ",
    ]
    # And one more with the same pattern as above...
    # SAIAMMMLM
    SAIAMMMLM_hits = [
        "SAIAMMMLMQQY",
        "FFSAIAMMMLMQQY",
        "FFFSAIAMMMLM"
    ]
    hits = SIINFEKLF_hits + SQIQQQQLQ_hits + SAIAMMMLM_hits
    _, result = bcp.fit_predict(hits)
    expected_binding_cores = {"SAIAMMMLM", "SQIQQQQLQ", "SIINFEKLF"}
    eq_(expected_binding_cores, set(result.values()))