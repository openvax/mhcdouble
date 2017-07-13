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


from pepnet import Encoder

from .dataset import Dataset
from .assembly import assemble_and_assign_to_sequence_groups
from .common import groupby
from .motifs import compatibility_score_for_binding_cores


from sklearn.linear_model import LogisticRegressionCV

class BindingCorePredictor(object):
    def __init__(
            self,
            encoder=Encoder(variable_length_sequences=False),
            length=9,
            model_fn=lambda: LogisticRegressionCV(penalty="l1", solver="liblinear"),
            initial_model=None):
        self.length = length
        self.encoder = encoder
        self.model_fn = model_fn
        self.model = initial_model

    def _build_training_set(
            self,
            sequence_groups):
        """
        Create an initial dataset for training a binding core predictor from
        overlapping peptides of variable length  by extracting all 9mers from the
        binding core region of assembled sequences. Any 9mers from outside
        the candidate region are retained as negative examples.
        """
        binding_core_training_peptides = []
        binding_core_training_labels = []
        binding_core_training_weights = []
        binding_core_training_group_ids = []
        k = self.length
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
            if n_neg > 0:
                binding_core_training_peptides.extend(negative_peptides)
                binding_core_training_labels.extend([False] * n_neg)
                binding_core_training_weights.extend([1.0 / n_neg] * n_neg)
                binding_core_training_group_ids.extend([group_id] * n_neg)
        return Dataset(
            alleles=None,
            peptides=binding_core_training_peptides,
            labels=binding_core_training_labels,
            weights=binding_core_training_weights,
            group_ids=binding_core_training_group_ids)


    def _find_binding_cores(self, candidate_dict):
        """
        Parameters
        ----------
        candidate_dict : dict
            Dictionary mapping group_id to a set of a peptides, each of which
            is a candidate binding core.

        Returns dictionary mapping each group_id to the highest scoring binding
        core.
        """
        result = {}
        for group_id, peptides in candidate_dict.items():
            result[group_id], _ = self.predict_best_peptide(peptides)
        return result

    def _restrict_dataset_to_best_binding_cores(self, dataset):
        positive_dataset = dataset[dataset.labels]
        negative_dataset = dataset[~dataset.labels]
        positive_peptide_groups = groupby(
            positive_dataset.peptides,
            keys=positive_dataset.group_ids)
        group_to_binding_core = self._find_binding_cores(positive_peptide_groups)
        n_kept = len(group_to_binding_core)
        positive_subset = Dataset(
            alleles=dataset.alleles[:n_kept],
            peptides=list(group_to_binding_core.values()),
            group_ids=list(group_to_binding_core.keys()))
        return negative_dataset.combine(
            positive_subset, preserve_group_ids=True)

    def fit_predict(self, hit_peptides, max_training_iters=100):
        """
        Given a set of MHCII ligands, assemble their sequences into longer
        contigs, infer candidate binding core regions, iteratively train
        a binding core classifier.
        Returns list of SequenceGroup objects (one for each contig) and
        dictionary mapping each contig to its predicted binding core.
        """
        sequence_groups = assemble_and_assign_to_sequence_groups(hit_peptides)
        self._fit_assembled_sequences(
            sequence_groups,
            max_training_iters=max_training_iters)
        contig_to_binding_core = self.predict_assembled_sequences(sequence_groups)
        return sequence_groups, contig_to_binding_core

    def _fit_assembled_sequences(self, sequence_groups, max_training_iters=100):
        """
        Iterative training of a fixed length binding core classifier which is
        then used to refine the training data.
        """
        dataset = self._build_training_set(sequence_groups)
        best_score = 0
        for i in range(max_training_iters):
            _, score = self._fit_iter(dataset)
            print("Iter #%d, PSSM score = %0.4f" % (i + 1, score))
            if score > best_score:
                best_score = score
            else:
                return score
        return best_score

    def _fit_iter(self, full_dataset):
        if self.model is None:
            # if a model hasn't been trained yet then just use the initial
            # training set where all candidate binding cores have equal weight
            current_dataset = full_dataset
        else:
            current_dataset = self._restrict_dataset_to_best_binding_cores(
                full_dataset)
        X, y, weights = self.encode_dataset(current_dataset)
        self.model = self.model_fn()
        self.model.fit(X, y, sample_weight=weights)

        selected_binding_cores = current_dataset[current_dataset.labels].peptides
        score = compatibility_score_for_binding_cores(selected_binding_cores)
        return selected_binding_cores, score


    def encode_dataset(self, dataset):
        return self.encode_arrays(
            peptides=dataset.peptides,
            labels=dataset.labels,
            weights=dataset.weights)

    def encode_arrays(self, peptides, labels, weights):
        X = self.encode_peptides(peptides)
        n_samples = len(X)
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
        X = self.encoder.encode_onehot(
            peptides,
            max_peptide_length=self.length)
        n_samples, expect_peptide_length, expect_n_amino_acids = X.shape
        if expect_peptide_length != self.length:
            raise ValueError("Expected 2nd dim to be %d but got %d" % (
                self.length, expect_peptide_length))
        if expect_n_amino_acids != 20:
            raise ValueError("Expected final dimension to be 20 but got %d" % (
                expect_n_amino_acids,))

        n_dims = self.length * 20
        X = X.reshape((n_samples, n_dims))
        assert len(X) == len(peptides), X.shape
        return X

    def predict_best_peptide(self, peptides):
        """
        Returns peptide among the inputs with the highest predicted
        binding core score (along with the score itself)
        """
        scores = self.predict_scores_for_peptides(peptides)
        idx = np.argmax(scores)
        return peptides[idx], scores[idx]

    def predict_labels_for_peptides(self, peptides):
        X = self.encode_peptides(peptides)
        assert self.model is not None
        return self.model.predict(X)

    def predict_scores_for_peptides(self, peptides):
        X = self.encode_peptides(peptides)
        assert self.model is not None
        return self.model.predict_proba(X)[:, -1]

    def predict_binding_core(self, sequence):
        """
        Given a long sequence, return binding core within it
        (along with corresponding score)
        """
        k = self.length
        peptides = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
        return self.predict_best_peptide(peptides)

    def predict_assembled_sequences(self, sequence_groups):
        result = {}
        for group in sequence_groups:
            binding_core_seq, _ = self.predict_binding_core(group.contig)
            result[group.contig] = binding_core_seq
        return result
