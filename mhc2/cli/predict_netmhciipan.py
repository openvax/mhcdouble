# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
import sys

from mhctools import NetMHCIIpan
import numpy as np
import pandas as pd

from .common import parse_args
from ..dataset import Dataset

parser = ArgumentParser(
    description="Run NetMHCIIpan over all peptides in sequence groups, compute AUC")

input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--dataset-csv")
input_group.add_argument("--hits-txt", nargs="+")


parser.add_argument("--output-csv", required=True)

def pad(peptide, min_length=9, padding_amino_acid="A"):
    n = len(peptide)
    if n >= min_length:
        return peptide
    n_padding = (min_length - n)
    padding_string = padding_amino_acid * n_padding
    return peptide + padding_string

def ensure_peptide_lengths(peptides, min_length=9):
    return [pad(p, min_length=min_length) for p in peptides]

def main(args_list=None):
    if not args_list:
        args_list = sys.argv[1:]
    args = parse_args(parser, args_list)
    print(args)
    if args.hits_txt and len(args.hits_txt) > 0:
        dataset = Dataset.from_peptides_text_files(args.hits_txt)
        dataset = dataset.assemble_contigs()
    else:
        dataset = Dataset.from_csv(args.dataset_csv)

    dataframes = []

    for allele, allele_dataset in dataset.groupby_allele():
        print("-- %s" % allele)
        df = allele_dataset.to_dataframe()
        netmhciipan = NetMHCIIpan([allele])

        # pad peptides with alanines in case they're shorter than 9mer
        peptides = ensure_peptide_lengths(df.peptide)
        n_peptides = len(peptides)
        binding_predictions = netmhciipan.predict_peptides(peptides)
        ic50_pred = np.array([x.affinity for x in binding_predictions])
        percentile_rank_pred = np.array([x.percentile_rank for x in binding_predictions])
        assert len(ic50_pred) == n_peptides
        assert len(percentile_rank_pred) == n_peptides
        df["netmhciipan_ic50"] = ic50_pred
        df["netmhciipan_percentile_rank"] = percentile_rank_pred
        dataframes.append(df)
    df = pd.concat(dataframes)
    df.to_csv(args.output_csv, index=False)
