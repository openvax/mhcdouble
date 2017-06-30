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

def filter_peptides(
        peptides,
        to_upper=True,
        exclude_set=set([]),
        invalid_amino_acids="*XU"):
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
