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
from ..peptides import load_allele_to_peptides_dict
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
parser.add_argument("--output-csv")
parser.add_argument("--output-min-binding-cores", default=1, type=int)


def main(args_list=None):
    args = parse_args(parser, args_list)
    print(args)
    if args.dataset_csv:
        dataset = Dataset.from_csv(args.dataset_csv)

        allele_to_hits = {}
        for allele, allele_dataset in dataset.groupby_allele():
            allele_to_hits[allele] = allele_dataset.peptides
    elif args.hits:
        allele_to_hits = load_allele_to_peptides_dict(args.hits)
    allele_to_groups = {}
    combined_list = []
    for (allele, hits) in allele_to_hits.items():

        hits = list(set(hit.upper() for hit in hits))
        n_hits = len(hits)
        print("Loaded %d peptides for allele %s" % (n_hits, allele))
        allele_groups = assemble_into_sequence_groups(hits)
        allele_groups = [
            g
            for g in allele_groups
            if len(g.binding_cores) >= args.output_min_binding_cores
        ]
        print_sequence_group_stats(allele_groups)
        allele_to_groups[allele] = allele_groups
        combined_list.extend(allele_groups)
    if args.output_txt_file:
        save_sequence_groups_to_txt_file(combined_list, path=args.output_txt_file)
    if args.output_json_file:
        save_sequence_groups_to_json_file(combined_list, path=args.output_json_file)
    if args.output_csv:
        dataset = Dataset.allele_to_sequence_groups_dict(allele_to_groups)
        dataset.to_csv(args.output_csv)
