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

from collections import defaultdict

from six import string_types
from glob import glob
from .mhc_names import parse_allele_from_filename

def filter_peptides(
        peptides,
        to_upper=True,
        exclude_set=set([]),
        invalid_amino_acids="*XU",
        remove_duplicates=True):
    """
    Strip and uppercase peptides (if necessary) and drop any
    empty sequences.
    """
    new_peptides = []
    for peptide in peptides:
        peptide = peptide.strip()
        if to_upper:
            peptide = peptide.upper()
        if not peptide or peptide in exclude_set or any(
                x in peptide for x in invalid_amino_acids):
            continue
        else:
            new_peptides.append(peptide)
    if remove_duplicates:
        new_peptides = list(set(new_peptides))
    return new_peptides

def load_peptides_list_from_path(path, to_upper=True):
    peptides = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            peptides.append(line.split()[0])
    return filter_peptides(peptides, to_upper=to_upper)

def load_path_to_peptides_dict(paths, to_upper=True, remove_duplicates=True):
    if isinstance(paths, string_types):
        paths = [paths]
    path_to_peptides_dict = {}
    for glob_or_path in paths:
        for path in glob.glob(glob_or_path):
            peptides = load_peptides_list_from_path(path)
            print("-- Loaded %d peptides from %s" % (
                len(peptides),
                path))
            path_to_peptides_dict[path] = peptides
    return path_to_peptides_dict

def load_allele_to_peptides_dict(paths, to_upper=True, remove_duplicates=True):
    path_to_peptides_dict = load_path_to_peptides_dict(
        paths,
        to_upper=to_upper,
        remove_duplicates=remove_duplicates)
    allele_to_peptides_dict = defaultdict(list)
    for (path, peptides) in path_to_peptides_dict.items():
        allele = parse_allele_from_filename(path)
        if not allele:
            raise ValueError("Could not determine allele name for '%s'" % path)
        allele_to_peptides_dict[allele].extend(peptides)
    counts = {
        allele: len(peptides)
        for (allele, peptides)
        in allele_to_peptides_dict.items()
    }
    print("Allele counts: %s" % (counts,))
    return allele_to_peptides_dict
