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
from ..peptides import filter_peptides, load_peptides_list

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
    help="Text file with a peptide on every line (excluding comment lines)")

parser.add_argument(
    "--input-csv",
    help=("""CSV with the following columns:
    - allele : string
    - peptide : string"""))

parser.add_argument(
    "--output",
    help="Name of CSV file which will contain results.")


def main(args_list=None):
    args = parse_args(parser, args_list)
    print("Args: %s" % (args,))
    alleles = args.allele
    if not alleles:
        raise ValueError("Expected at least one HLA Class II allele")

    peptides = []
    if args.peptide:
        peptides.extend(filter_peptides(args.peptide))
    if args.peptides_file:
        peptides.extend(load_peptides_list(args.peptides_file))
    model_collection = ModelCollection(args.model_dir)
    partial_result_dataframes = []

    if len(peptides) > 0:
        for allele in alleles:
            model = model_collection[allele]
            partial_result_dataframes.append(
                model.predict_dataframe(peptides))
    if args.input_csv:
        input_df = pd.read_csv(args.input_csv)
        for allele, group in input_df.groupby("allele"):
            input_peptides = list(group["peptide"])
            partial_result_dataframes.append(
                model.predict_dataframe(input_peptides))
    if len(partial_result_dataframes) == 0:
        raise ValueError("No peptides given!")
    result_df = pd.concat(partial_result_dataframes)
    print(result_df)
    if args.output:
        result_df.to_csv(args.output, index=False)

