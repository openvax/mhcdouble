from mhc2.fixed_length_predictor import FixedLengthPredictor
import numpy as np

def test_fixed_length_predictor():
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
    SQIAMMRLR_hits = ["SQIAMMRLR", "F" * 5 + "SQIAMMRLR"]
    SMIMMMQLQ_hits = ["SMIMMMQLQ", "SMIMMMQLQ" + 7 * "A"]
    SFIFFFILI_hits = ["SFIFFFILI", "Q" * 3 + "SFIFFFILI"]
    hits = (
        SIINFEKLF_hits + SQIQQQQLQ_hits + SAIAMMMLM_hits + SQIAMMRLR_hits +
        SMIMMMQLQ_hits + SFIFFFILI_hits)
    predictor = FixedLengthPredictor()
    y = predictor.fit_predict_peptides(hits)
    assert np.all(y > 0.5)
