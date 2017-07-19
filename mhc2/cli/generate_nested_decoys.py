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
from ..sequence_group import (
    save_sequence_groups_to_txt_file,
    flatten_sequence_groups
)


parser = ArgumentParser(description="Extend a dataset with decoys")

input_group = parser.add_mutually_exclusive_group(required=True)

input_group.add_argument(
    "--hits-txt",
    help="Text file containing one peptide per line",
    nargs="+")

input_group.add_argument(
    "--hits-csv",
    help="CSV file containing a preprocessed dataset")


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

parser.add_argument("--output-txt")

def extract_binding_core_subsequences(sequence_groups, k=7):
    result = set([])
    for g in sequence_groups:
        for c in g.binding_cores:
            for i in range(len(c) - k + 1):
                result.add(c[i:i + k])
    return result

def main(args_list=None):
    args = parse_args(parser, args_list)
    print(args)
    if args.hits_csv:
        dataset = Dataset.from_csv(args.hits_csv)
    elif args.hits_txt:
        dataset = Dataset.from_peptides_text_files(args.hits_txt)
    else:
        raise ValueError(
            "Specify hits via --hits-csv or --hits-txt")
    allele_to_decoy_sequence_groups = {}
    decoy_datasets = []
    for (allele, dataset) in dataset.groupby_allele():
        sequence_groups = dataset.to_sequence_groups()
        n_hit_loci = len(sequence_groups)
        binding_core_subsequences = extract_binding_core_subsequences(
            sequence_groups)
        peptides, _, _, _ = flatten_sequence_groups(sequence_groups)
        print("Assembled %d hit peptides for allele %s into %d loci" % (
            len(peptides),
            allele,
            n_hit_loci))
        n_decoy_loci = int(args.decoys_per_hit * n_hit_loci)
        print("Generating %d decoy loci for allele %s" % (n_decoy_loci, allele))
        decoys = generate_decoy_sequence_groups(
            n_decoy_loci=n_decoy_loci,
            min_decoys_per_locus=args.min_peptides_per_locus,
            max_decoys_per_locus=args.max_peptides_per_locus,
            binding_core_length=args.binding_core_length,
            contig_length=args.contig_length,
            exclude_subsequences=binding_core_subsequences)
        decoy_dataset = Dataset.from_sequence_groups(
            allele=allele,
            label=False,
            sequence_groups=decoys)
        decoy_datasets.append(decoy_dataset)
        allele_to_decoy_sequence_groups[allele] = decoys

    if args.output_txt:
        all_sequence_groups = []
        for _, sg in allele_to_decoy_sequence_groups.items():
            all_sequence_groups.extend(sg)
        save_sequence_groups_to_txt_file(all_sequence_groups, path=args.output_txt)

    if args.output_csv:
        combined_dataset = Dataset.concat(decoy_datasets)
        combined_dataset.to_csv(args.output_csv)
