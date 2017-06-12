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

import pandas as pd

from .common import parse_args
from ..model_collection import ModelCollection

parser = ArgumentParser(description="Predict peptide-MHC Class II binding")

parser.add_argument(
    "--model-dir",
    help="Directory containing trained MHC II model",
    required=True)

parser.add_argument(
    "--allele",
    nargs="+",
    help=(
        "Class II MHC allele(s), alpha chain can be omitted for HLA-DR. "
        "You can run `mhc2-alleles model-dir/` to see which alleles are "
        "available for a particular model."))

parser.add_argument("--peptide", nargs="*", help="Peptide sequence")

parser.add_argument(
    "--peptides-file",
    help="Text file with a peptide on every line (excluding comment lines)",
    nargs="*",
    default=[])

parser.add_argument(
    "--input-csv",
    help=("""CSV with the following columns:
    - allele : string
    - peptide : string"""))

parser.add_argument(
    "--output",
    help="Name of CSV file which will contain results.")

def filter_peptides(peptides, to_upper=True):
    """
    Strip and uppercase peptides (if necessary) and drop any
    empty sequences.
    """
    new_peptides = []
    for peptide in peptides:
        peptide = peptide.strip()
        if to_upper:
            peptide = peptide.upper()
        if not peptide:
            continue
        else:
            new_peptides.append(peptide)
    return new_peptides


def load_peptides_list(path, to_upper=True):
    peptides = []
    with open(path) as f:
        for line in f:
            if f.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            peptides.append(line.split()[0])
    return filter_peptides(peptides, to_upper=to_upper)

def main(args_list=None):
    args = parse_args(parser, args_list)
    print("Args: %s" % (args,))
    alleles = args.allele
    if not alleles:
        raise ValueError("Expected at least one HLA Class II allele")
    peptides = []
    peptides.extend(filter_peptides(args.peptide))
    for path in args.peptides_file:
        peptides.extend(load_peptides_list(path))

    model_collection = ModelCollection(args.model_dir)

    partial_result_dataframes = []
    for allele in alleles:
        model = model_collection[allele]
        partial_result_dataframes.append(
            model.predict_dataframe(peptides))
    result_df = pd.concat(partial_result_dataframes)
    print(result_df)
    if args.output:
        result_df.to_csv(args.output, index=False)

