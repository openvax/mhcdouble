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

import argparse

import numpy as np

from .common import parse_args
from ..dataset import Dataset
from ..peptides import load_peptides_list_from_path
from ..assembly import split_into_sequence_groups

parser = argparse.ArgumentParser()
input_group = parser.add_mutually_exclusive_group(required=True)

input_group.add_argument("--dataset-csv")
input_group.add_argument("--hits")

def main(args_list=None):
    args = parse_args(parser, args_list)
    print(args)
    if args.dataset_csv:
        dataset = Dataset.from_csv(args.dataset_csv)
        hits = dataset[dataset.labels].peptides
    elif args.hits:
        hits = load_peptides_list_from_path(args.hits)
    hits = list(set(hit.upper() for hit in hits))
    n_hits = len(hits)
    print("Loaded %d peptides" % n_hits)
    groups = split_into_sequence_groups(hits)
    print("# groups = %d" % len(groups))
    children_per_group = [len(g.children) for g in groups]
    print("# sequences per group min = %d, mean = %f, max = %d (%s)" % (
        min(children_per_group),
        np.mean(children_per_group),
        max(children_per_group),
        max(groups, key=lambda g: len(g.children) == max(children_per_group)).contig))
    leaves_per_group = [len(g.leaves) for g in groups]
    print("# leaves per group min = %d, mean = %f, max = %d (%s)" % (
        min(leaves_per_group),
        np.mean(leaves_per_group),
        max(leaves_per_group),
        max(groups, key=lambda g: len(g.leaves) == max(leaves_per_group)).contig))
    binding_cores_per_group = [len(g.binding_cores) for g in groups]
    print("# binding cores per group min = %d, mean = %f, max = %d (%s)" % (
        min(binding_cores_per_group),
        np.mean(binding_cores_per_group),
        max(binding_cores_per_group),
        max(groups, key=lambda g: len(g.binding_cores) == max(binding_cores_per_group)).contig))
    child_to_binding_core_ratios = [len(g.children) / len(g.binding_cores)  for g in groups]
    print("Ratio of children to binding cores min = %f, mean = %f, max = %f" % (
        np.min(child_to_binding_core_ratios),
        np.mean(child_to_binding_core_ratios),
        np.max(child_to_binding_core_ratios)))

