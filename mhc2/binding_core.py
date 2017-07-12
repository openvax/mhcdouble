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

import numpy as np

from .dataset import Dataset
from .assembly import split_into_sequence_groups

from pepnet import Encoder

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


class BindingCorePredictor(object):
    def __init__(self, model, length=9):
        self.length = length
        self.encoder = Encoder(variable_length_sequences=False)
        self.model = None

    def fit_dataset(self):
        pass

    def fit(self):
        pass

    def encode_dataset(self, dataset):
        return self.encode(
            peptides=dataset.peptides,
            labels=dataset.labels,
            weights=dataset.weights)

    def encode_arrays(self, peptides, labels, weights):
        X = self.encode_peptides(peptides)
        n_samples, expect_peptide_length, expect_n_amino_acids = X.shape
        if expect_peptide_length != self.length:
            raise ValueError("Expected 2nd dim to be %d but got %d" % (
                self.length, expect_peptide_length))
        if expect_n_amino_acids != 20:
            raise ValueError("Expected final dimension to be 20 but got %d" % (
                expect_n_amino_acids,))
        X = X.reshape((n_samples, self.length * 20))
        y = np.array(labels)
        assert n_samples == len(y), len(y)
        weights = np.array(weights)
        return X, y, weights

    def encode_peptides(self, peptides):
        if any(len(p) != self.length for p in peptides):
            raise ValueError(
                "Expected all peptides to be %d in length, got %s" % (
                    self.length,
                    {len(p) for p in peptides}))
        return self.encoder.encode_onehot(
            peptides,
            max_peptide_length=self.length)


    def predict(self, peptides):
        X = self.encode_peptides(peptides)
        assert self.model is not None
        return self.model.predict(X)

    def predict_proba(self, peptides):
        X = self.encode_peptides(peptides)
        assert self.model is not None
        return self.model.predict_proba(X)[:, -1]

def assemble_and_identify_binding_cores(
        hits,
        binding_core_size=9,
        allele_name=None):
    sequence_groups = split_into_sequence_groups(hits)
    dataset = build_initial_training_set(
        sequence_groups,
        binding_core_size=binding_core_size,
        allele_name=allele_name)


def find_binding_cores(binding_core_predict, group_id_to_candidate_seqs_dict):

    kept_positive_group_ids = []
    kept_positive_peptides = []
    for group_id, peptides in group_ids_to_peptides.items():
        X_group = self.encode(oeot)
        n_dims = k * 20
        X_group = X_group.reshape((len(peptides), n_dims))
        assert len(X_group) == len(peptides), X_group.shape
        pred = clf.predict_proba(X_group)[:, -1]

        max_idx = np.argmax(pred)

        kept_positive_group_ids.append(group_id)
        kept_positive_peptides.append(peptides[max_idx])
    n_kept = len(kept_positive_peptides)
    positive_subset = Dataset(
        alleles=dataset.alleles[:n_kept],
        peptides=kept_positive_peptides,
        group_ids=kept_positive_group_ids)
    return negative_dataset.combine(positive_subset, preserve_group_ids=True)

for (peptide, group_id) in zip(positive_dataset.peptides, positive_dataset.group_ids):
    group_ids_to_peptides[group_id].append(peptide)


positive_dataset = dataset[dataset.labels]
negative_dataset = dataset[~dataset.labels]
group_ids_to_peptides = defaultdict(list)

