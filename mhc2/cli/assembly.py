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


from .common import parse_args
from ..dataset import Dataset
from ..peptides import load_peptides_list_from_path
from ..assembly import assemble_into_sequence_groups
from ..sequence_group import (
    print_sequence_group_stats,
    save_sequence_groups_to_txt_file,
    save_sequence_groups_to_json_file
)

parser = argparse.ArgumentParser()
input_group = parser.add_mutually_exclusive_group(required=True)

input_group.add_argument("--dataset-csv")
input_group.add_argument("--hits")

parser.add_argument("--output-txt-file")
parser.add_argument("--output-json-file")
parser.add_argument("--output-min-binding-cores", default=1, type=int)


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
    groups = assemble_into_sequence_groups(hits)
    print_sequence_group_stats(groups)
    groups = [
        g
        for g in groups
        if len(g.binding_cores) < args.output_min_binding_cores
    ]
    if args.output_txt_file:
        save_sequence_groups_to_txt_file(groups, path=args.output_txt_file)
    if args.output_json_file:
        save_sequence_groups_to_json_file(groups, path=args.output_json_file)
