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

from .common import parse_args
from ..dataset import Dataset
from ..decoys import generate_decoy_sequence_groups
from ..peptides import load_peptides_list_from_path
from ..assembly import assemble_into_sequence_groups
from ..sequence_group import (
    load_sequence_groups_from_json_file,
    save_sequence_groups_to_json_file,
    save_sequence_groups_to_txt_file,
    flatten_sequence_groups
)


parser = ArgumentParser(description="Extend a dataset with decoys")

input_group = parser.add_mutually_exclusive_group(required=True)

input_group.add_argument(
    "--hits-txt",
    help="Text file containing one peptide per line")

input_group.add_argument(
    "--hits-csv",
    help="CSV file containing a preprocessed dataset")

input_group.add_argument(
    "--hits-json",
    help="JSON representation of sequence groups")

parser.add_argument(
    "--decoys-per-hit",
    type=float,
    default=10.0,
    help="Number of decoy loci per hit locus")

parser.add_argument("--min-peptides-per-locus", type=int, default=1)
parser.add_argument("--max-peptides-per-locus", type=int, default=10)
parser.add_argument("--min-contig-length", type=int, default=25)
parser.add_argument("--max-contig-length", type=int, default=65)
parser.add_argument("--binding-core-length", type=int, default=9)
parser.add_argument("--contig-length", type=int, default=40)

parser.add_argument(
    "--output-csv",
    help="Name of output CSV for decoy dataset")

parser.add_argument(
    "--output-json",
    help="Name of output JSON for decoy sequence groups")

parser.add_argument("--output-txt")

def main(args_list=None):
    args = parse_args(parser, args_list)
    print(args)
    if args.hits_csv:
        dataset = Dataset.from_csv(args.hits_csv)
        hits = dataset[dataset.labels].peptides
        sequence_groups = assemble_into_sequence_groups(hits)
    elif args.hits_txt:
        hits = load_peptides_list_from_path(args.hits_txt)
        sequence_groups = assemble_into_sequence_groups(hits)
    elif args.hits_json:
        sequence_groups = load_sequence_groups_from_json_file(args.hits_json)
    else:
        raise ValueError(
            "Specify hits via --hits-csv, --hits-txt, --hits-json")
    n_hit_loci = len(sequence_groups)
    peptides, _ = flatten_sequence_groups(sequence_groups)
    print("Assembled %d hit peptides into %d loci" % (
        len(peptides),
        n_hit_loci))
    n_decoy_loci = int(args.decoys_per_hit * n_hit_loci)
    print("Generating %d decoy loci" % n_decoy_loci)
    decoys = generate_decoy_sequence_groups(
        n_decoy_loci=n_decoy_loci,
        min_decoys_per_locus=args.min_peptides_per_locus,
        max_decoys_per_locus=args.max_peptides_per_locus,
        binding_core_length=args.binding_core_length,
        contig_length=args.contig_length)
    if args.output_txt:
        save_sequence_groups_to_txt_file(decoys, path=args.output_txt)
    if args.output_json:
        save_sequence_groups_to_json_file(decoys, path=args.output_json)
    if args.output_csv:
        peptides = []
        group_ids = []
        for i, g in sequence_groups:
            for p in g.children:
                peptides.append(p)
                group_ids.append(i)
        with open(args.output_csv, "w") as f:
            f.write("peptide,group_id\n")
            for p, g_id in zip(peptides, group_ids):
                f.write("%s,%d\n" % (p, g_id))

