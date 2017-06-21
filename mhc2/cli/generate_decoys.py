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
from ..dataset import Dataset

parser = ArgumentParser(description="Extend a dataset with decoys")
parser.add_argument(
    "--input-csv",
    help=("""Name of input CSV containing mass spec hits, expected to have the following columns:
    - "allele" : string (can also be "mhc")
    - "peptide" : string (can also be "seq")"""),
    required=True)

parser.add_argument(
    "--output-csv",
    help=("""Name of output CSV file which will have the following columns:
    - "allele" : string
    - "peptide" : string
    - "label" : bool"""),
    required=True)

def main(args_list=None):
    args = parse_args(parser, args_list)
    dataset = Dataset.from_csv(args.input_csv)
    dataset.to_csv(args.output_csv)