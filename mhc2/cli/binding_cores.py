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

import sys
from argparse import ArgumentParser

import numpy as np

from .common import parse_args

from ..peptides import load_peptides_list_from_path
from ..binding_core_predictor import BindingCorePredictor

parser = ArgumentParser(
    description="Iterative training of binding-core predictor and overall predictor")

parser.add_argument("--hits", required=True)

parser.add_argument("--binding-core-length", type=int, default=9)

parser.add_argument(
    "--save-binding-core-training-csv",
    help="Optional path to CSV which saves training data for binding core predictor")

parser.add_argument(
    "--prob",
    type=float,
    default=0.05,
    help="Prior probability that a random sequence is a binding core")

def main(args_list=None):
    if not args_list:
        args_list = sys.argv[1:]
    args = parse_args(parser, args_list)
    hits = load_peptides_list_from_path(args.hits)
    bcp = BindingCorePredictor(length=args.binding_core_length)
    sequence_groups, contig_to_predictions = bcp.fit_predict(hits)
    n_correct = 0
    n_total = 0
    for group in sequence_groups:
        subsequences, scores = contig_to_predictions[group.contig]
        maxidx = np.argmax(scores)

        predicted_binding_core = subsequences[maxidx]
        correct = any(
            predicted_binding_core in candidate_region
            for candidate_region in group.binding_cores)
        n_correct += correct
        n_total += 1
    print("# correct = %d/%d" % (n_correct, n_total))
