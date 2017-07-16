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
from collections import defaultdict
import glob

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from .common import parse_args

from ..peptides import load_peptides_list_from_path
from ..fixed_length_predictor import FixedLengthPredictor
from ..decoys import generate_decoy_sequence_groups
from ..assembly import assemble_and_assign_to_sequence_groups

parser = ArgumentParser(
    description="Iterative training of binding-core predictor and overall predictor")

parser.add_argument("--hits", required=True, nargs="+")

parser.add_argument("--training-decoys-per-hit", type=int, default=10)

parser.add_argument("--binding-core-length", type=int, default=9)

parser.add_argument("--allele", help="Allele name to use in training datasets")

parser.add_argument(
    "--save-binding-core-training-csv",
    help="Optional path to CSV which saves training data for binding core predictor")

parser.add_argument("--max-binding-core-iters", type=int, default=100)
parser.add_argument("--test-decoys-per-hit", type=int, default=100)
parser.add_argument("--folds", type=int, default=3)
parser.add_argument("--nterm-residues", type=int, default=1)
parser.add_argument("--cterm-residues", type=int, default=1)
parser.add_argument("--prediction-binding-cores", type=int, default=1)
parser.add_argument("--training-binding-cores", type=int, default=1)


def main(args_list=None):
    if not args_list:
        args_list = sys.argv[1:]
    args = parse_args(parser, args_list)
    print(args)

    allele_to_aucs_dict = defaultdict(list)

            hit_sequence_groups = assemble_and_assign_to_sequence_groups(hits)
            n_hit_loci = len(hit_sequence_groups)
            kfold = KFold(args.folds, shuffle=True)
            training_hit_loci_per_fold = int(
                ((args.folds - 1.0) / args.folds) * n_hit_loci)
            n_decoy_loci = 2 * training_hit_loci_per_fold * args.training_decoys_per_hit
            decoy_sequence_groups = generate_decoy_sequence_groups(n_decoy_loci=n_decoy_loci)
            training_decoy_sequence_groups = decoy_sequence_groups[:n_decoy_loci // 2]
            test_decoy_sequence_groups = decoy_sequence_groups[n_decoy_loci // 2:]

            for i, (train_idx, test_idx) in enumerate(kfold.split(hit_sequence_groups)):
                train_hit_groups = [hit_sequence_groups[i] for i in train_idx]
                test_hit_groups = [hit_sequence_groups[i] for i in test_idx]
                pred = FixedLengthPredictor(
                    binding_core_size=args.binding_core_length,
                    max_binding_core_iters=args.max_binding_core_iters,
                    n_nterm=args.nterm_residues,
                    n_cterm=args.cterm_residues,
                    n_binding_cores_train=args.prediction_binding_cores,
                    n_binding_cores_predict=args.training_binding_cores)
                pred.fit_sequence_groups(
                    train_hit_groups,
                    training_decoy_sequence_groups)
                y_pos = pred.predict_sequence_groups(test_hit_groups)
                y_neg = pred.predict_sequence_groups(test_decoy_sequence_groups)
                y_score = np.concatenate([y_pos, y_neg])
                y_true = np.array([True] * len(y_pos) + [False] * len(y_neg))
                auc = roc_auc_score(y_true=y_true, y_score=y_score)
                print("Fold %d/%d allele=%s AUC=%0.4f" % (
                    i + 1, args.folds, allele, auc))
                allele_to_aucs_dict[allele].append(auc)
    if len(allele_to_aucs_dict) > 0:
        sum_aucs = 0
        count = 0
        for allele, aucs in allele_to_aucs_dict.items():
            print("-- %s mean AUC=%0.4f std=%0.4f" % (
                allele, np.mean(aucs), np.std(aucs)))
            sum_aucs += np.sum(aucs)
            count += len(aucs)
        print("Overall mean AUC=%0.4f" % (sum_aucs / count))
