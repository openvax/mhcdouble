from nose.tools import eq_
from mhc2 import Dataset
import pandas as pd

def test_from_dataframe():
    expected_dataset = Dataset(
        alleles=["DRB1*01:01"],
        peptides="SIINFEKL",
        labels=[True])
    for peptide_column_name in ["peptide", "peptides", "seq", "seqs"]:
        for mhc_column_name in ["mhc", "allele", "alleles"]:
            for label_column_name in ["hit", "label", "hits", "labels"]:
                df = pd.DataFrame({
                    peptide_column_name: ["SIINFEKL"],
                    mhc_column_name: ["DRB1*01:01"],
                    label_column_name: [True]})
                dataset = Dataset.from_dataframe(df)
                eq_(dataset, expected_dataset)