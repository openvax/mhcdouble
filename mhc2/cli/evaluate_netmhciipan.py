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

from .common import parse_args
from ..sequence_group import (
    load_sequence_groups_from_json_file,
    flatten_sequence_groups
)
from ..peptides import load_peptides_list_from_path
from ..assembly import assemble_into_sequence_groups


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
    ok_indices = [i for i, p in enumerate(hit_peptides) if len(p) >= 9]
    hit_peptides = [hit_peptides[i] for i in ok_indices]
    hit_group_ids = [hit_group_ids[i] for i in ok_indices]

    netmhciipan = NetMHCIIpan([args.allele])

    print("Running NetMHCIIpan on %d hits" % len(hit_peptides))
    hit_pred = netmhciipan.predict_peptides(hit_peptides)
    print(hit_pred.mean())

    decoy_groups = load_sequence_groups_from_json_file(args.decoys_json)
    decoy_peptides, decoy_group_ids = flatten_sequence_groups(decoy_groups)
    print("Running NetMHCIIpan on %d decoys" % len(decoy_peptides))

    decoy_pred= netmhciipan.predict_peptides(decoy_peptides)
    print(decoy_pred.mean())
