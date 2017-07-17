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

from .common import parse_args
from ..sequence_group import (
    load_sequence_groups_from_json_file,
    flatten_sequence_groups
)
from ..peptides import load_peptides_list_from_path
from ..assembly import assemble_into_sequence_groups
from ..common import groupby_array_fn

from sklearn.metrics import roc_auc_score

parser = ArgumentParser(
    description="Run NetMHCIIpan over all peptides in sequence groups, compute AUC")

input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--hits-json")
input_group.add_argument("--hits-txt")

parser.add_argument("--decoys-json")
parser.add_argument("--allele", default="DRB1*01:01")

def main(args_list=None):
    if not args_list:
        args_list = sys.argv[1:]
    args = parse_args(parser, args_list)
    print(args)
    if args.hits_json:
        hit_groups = load_sequence_groups_from_json_file(args.hits_json)
    else:
        hit_peptides = load_peptides_list_from_path(args.hits_txt)
        hit_groups = assemble_into_sequence_groups(hit_peptides)
    hit_peptides, hit_group_ids = flatten_sequence_groups(hit_groups)
    max_hit_group_id = max(hit_group_ids)
    ok_indices = [i for i, p in enumerate(hit_peptides) if len(p) >= 9]
    hit_peptides = [hit_peptides[i] for i in ok_indices]
    hit_group_ids = [hit_group_ids[i] for i in ok_indices]

    netmhciipan = NetMHCIIpan([args.allele])

    print("Running NetMHCIIpan on %d hits" % len(hit_peptides))
    hit_binding_predictions = netmhciipan.predict_peptides(hit_peptides)

    def harmonic_mean_affinity(binding_predictions, eps=0.001):
        affinities = [x.affinity + eps for x in binding_predictions]
        log_aff = np.log(affinities)
        return np.exp(np.mean(log_aff))

    hit_pred = groupby_array_fn(
        hit_binding_predictions,
        hit_group_ids,
        size=max_hit_group_id + 1,
        fn=harmonic_mean_affinity) # lambda xs: np.mean([x.percentile_rank for x in xs]))
    assert len(hit_peptides) == len(hit_binding_predictions)
    assert all([p == x.peptide for (p, x) in zip(hit_peptides, hit_binding_predictions)])
    print("Min/Mean/Max Affinity: %0.4f/%0.4f/%0.4f" % (
        np.min(hit_pred),
        np.mean(hit_pred),
        np.max(hit_pred)))
    decoy_groups = load_sequence_groups_from_json_file(args.decoys_json)
    decoy_groups = decoy_groups[:1000]
    decoy_peptides, decoy_group_ids = flatten_sequence_groups(decoy_groups)

    print("Running NetMHCIIpan on %d decoys" % len(decoy_peptides))
    decoy_binding_predictions = netmhciipan.predict_peptides(decoy_peptides)

    assert len(decoy_peptides) == len(decoy_binding_predictions)
    assert all([p == x.peptide for (p, x) in zip(decoy_peptides, decoy_binding_predictions)])

    decoy_pred = groupby_array_fn(
        decoy_binding_predictions,
        decoy_group_ids,
        size=len(decoy_groups) + 1,
        fn=harmonic_mean_affinity) # lambda xs: np.mean([x.percentile_rank for x in xs]))

    print("Min/Mean/Max Affinity: %0.4f/%0.4f/%0.4f" % (
        np.min(decoy_pred),
        np.mean(decoy_pred),
        np.max(decoy_pred)))

    y_score = np.concatenate([hit_pred, decoy_pred])
    y_true = np.array([True] * len(hit_pred) + [False] * len(decoy_pred))
    auc = roc_auc_score(y_true=y_true, y_score=-y_score)
    print("AUC = %0.4f" % auc)

