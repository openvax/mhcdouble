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

"""
This module is for identifying binding cores from longer class II ligands
without using a previosly trained p-MHCII binding predictive model.
"""

from .dataset import Dataset
from .assembly import split_into_sequence_groups

def build_initial_training_set(
        sequence_groups,
        binding_core_size=9,
        allele_name=None):
    """
    Create an initial dataset for training a binding core predictor from
    overlapping peptides of variable length  by extracting all 9mers from the
    binding core region of assembled sequences.
    """
    binding_core_training_peptides = []
    binding_core_training_labels = []
    binding_core_training_weights = []
    binding_core_training_group_ids = []
    k = binding_core_size
    for group_id, g in enumerate(sequence_groups):
        if len(g.children) < 2:
            continue
        # for now we're going to ignore sequences with multiple candidate
        # binding cores, we should eventually either lift this restriction
        # or refine the binding core finding algorithm to make multiple
        # sequences rare
        if len(g.binding_cores) > 1:
            continue

        binding_core_region = g.binding_cores[0]
        print(binding_core_region)
        if len(binding_core_region) < k:
            print("Skipping short binding core (len=%d): %s from %s" % (
                len(binding_core_region),
                binding_core_region,
                g))
            continue

        positive_peptides = set([])
        for i in range(len(binding_core_region) - k + 1):
            positive_peptides.add(binding_core_region[i:i + k])

        n_pos = len(positive_peptides)
        binding_core_training_peptides.extend(positive_peptides)
        binding_core_training_labels.extend([True] * n_pos)
        binding_core_training_weights.extend([1.0 / n_pos] * n_pos)
        binding_core_training_group_ids.extend([group_id] * n_pos)
        full_sequence = g.contig
        negative_peptides = []

        for i in range(len(full_sequence) - k + 1):
            seq = full_sequence[i:i + k]
            if seq not in positive_peptides:
                negative_peptides.append(seq)
        n_neg = len(negative_peptides)
        binding_core_training_peptides.extend(negative_peptides)
        binding_core_training_labels.extend([False] * n_neg)
        binding_core_training_weights.extend([1.0 / n_neg] * n_neg)
        binding_core_training_group_ids.extend([group_id] * n_neg)
    return Dataset(
        alleles=([allele_name] * len(binding_core_training_peptides) if allele_name else None),
        peptides=binding_core_training_peptides,
        labels=binding_core_training_labels,
        weights=binding_core_training_weights,
        group_ids=binding_core_training_group_ids)

def assemble_and_identify_binding_cores(
        hits,
        binding_core_size=9,
        allele_name=None):
    sequence_groups = split_into_sequence_groups(hits)
    dataset = build_initial_training_set(
        sequence_groups,
        binding_core_size=binding_core_size,
        allele_name=allele_name)
