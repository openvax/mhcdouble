from nose.tools import eq_
from mhc2 import Dataset, SequenceGroup
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

def test_concat():
    expected_dataset = Dataset(
        alleles=["DRB1*01:01", "DRB1*01:02", "DRB1*01:02"],
        peptides=["SIINFEKL", "QSIINFEKL", "QQSIINFEKL"],
        labels=[True, True, False],
        weights=[1.0, 2.0, 0.1])
    split_datasets = [
        Dataset(alleles=["DRB1*01:01"], peptides=["SIINFEKL"], labels=[True], weights=[1.0]),
        Dataset(alleles=["DRB1*01:02"], peptides=["QSIINFEKL"], labels=[True], weights=[2.0]),
        Dataset(alleles=["DRB1*01:02"], peptides=["QQSIINFEKL"], labels=[False], weights=[0.1])
    ]
    combined = Dataset.concat(split_datasets)
    eq_(expected_dataset, combined)


contig = "DEVIGQVLSTLKSEDVPYTAALTAVRPSRVARDVA"
sequence_groups = [
    SequenceGroup(
        contig=contig,
        children=[
            "DEVIGQVLSTLKSEDVPYTAALTAVRPSRV",
                      "LKSEDVPYTAALTAVRPSRVARDVA",
                            "PYTAALTAVR",
                           "VPYTAALTAV",
        ],
        leaves={"PYTAALTAVR", "VPYTAALTAV"},
        binding_cores={"PYTAALTAV"})
]

def test_dataset_from_sequence_groups():
    dataset_from_sequence_groups = Dataset.from_sequence_groups(sequence_groups)
    peptides = dataset_from_sequence_groups.peptides
    dataset_from_peptides = Dataset(peptides=peptides)
    dataset_assembled = dataset_from_peptides.assemble_contigs()
    eq_(dataset_from_sequence_groups, dataset_assembled)
    eq_(dataset_from_sequence_groups.unique_alleles(), {None})

def test_dataset_from_sequence_groups_with_allele():
    dataset_from_sequence_groups = Dataset.from_sequence_groups(
        allele="HLA-A*02:01",
        sequence_groups=sequence_groups)
    eq_(dataset_from_sequence_groups.unique_alleles(), {"HLA-A*02:01"})

def test_dataset_shuffle():
    dataset = Dataset(
        alleles=["HLA-A0201"] * 20 + ["H-2-Kd"] * 20,
        peptides=["A" * 9] * 40)
    shuffled = dataset.shuffle()
    eq_(set(shuffled.alleles[:20]), {"HLA-A*02:01", "H-2-Kd"})

