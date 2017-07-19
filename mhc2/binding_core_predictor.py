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

from collections import defaultdict

import numpy as np


from pepnet import Encoder

from .dataset import Dataset
from .assembly import assemble_into_sequence_groups
from .common import groupby
# from .motifs import compatibility_score_for_binding_cores

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


class BindingCorePredictor(object):
    def __init__(
            self,
            encoder=Encoder(variable_length_sequences=False),
            length=9,
            predictor_class=LogisticRegression,
            lr_penalty="l2",
            lr_solver="liblinear",
            predictor_kwargs={},
            prior_prob=0.05,
            model_scoring_fn=roc_auc_score,
            n_cv_folds=3,
            max_iters=100):
        self.length = length
        self.encoder = encoder
        self.predictor_class = predictor_class
        self.predictor_kwargs = predictor_kwargs
        self.model_scoring_fn = model_scoring_fn
        self.n_cv_folds = n_cv_folds
        self.prior_prob = prior_prob
        self.model = None
        self.max_iters = max_iters
        self.lr_penalty = lr_penalty
        self.lr_solver = lr_solver

    def _make_model(self, C):
        return self.predictor_class(
            C=C,
            class_weight={0: 1.0 / self.prior_prob, 1: 1.0},
            penalty=self.lr_penalty,
            solver=self.lr_solver)

    def _cv_fit(self, dataset, Cs=15, min_C_exponent=-2, max_C_exponent=2):
        X, y, weights, group_ids = self.encode_dataset(dataset)
        C_values =  10.0 ** np.linspace(min_C_exponent, max_C_exponent, Cs)
        C_to_scores = defaultdict(list)
        group_kfold = GroupKFold(self.n_cv_folds)
        for (train_idx, test_idx) in group_kfold.split(X, y, group_ids):
            X_train, y_train, weights_train = X[train_idx], y[train_idx], weights[train_idx]
            X_test, y_test, weights_test = X[test_idx], y[test_idx], weights[test_idx]
            for C in C_values:
                model = self._make_model(C=C)
                model.fit(X_train, y_train, sample_weight=weights_train)
                y_pred = model.predict_proba(X_test)[:, -1]
                auc = roc_auc_score(
                    y_true=y_test,
                    y_score=y_pred,
                    sample_weight=weights_test)
                C_to_scores[C].append(auc)
        C_to_average_score = {C: np.mean(scores) for (C, scores) in C_to_scores.items()}
        best_C, best_score = max(C_to_average_score.items(), key=lambda x: x[1])
        print("[BindingCorePredictor] Selected model with C=%0.4f (AUC=%0.4f)" % (best_C, best_score))
        model = self._make_model(C=best_C)
        model.fit(X, y, sample_weight=weights)
        print("[BindingCorePredictor] Sparsity = %0.4f" % (np.mean(np.isclose(model.coef_.flatten(), 0))))
        return model, best_score, best_C

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
        binding_core_training_contigs = []
        k = self.length
        for g in sequence_groups:
            # for now we're going to ignore short binding cores and sequences with
            # multiple candidate binding cores
            binding_cores = [c for c in g.binding_cores if len(c) >= k]

            if len(binding_cores) != 1:
                continue

            positive_peptides = set([])

            binding_core_region = binding_cores[0]

            for i in range(len(binding_core_region) - k + 1):
                positive_peptides.add(binding_core_region[i:i + k])

            n_pos = len(positive_peptides)
            binding_core_training_peptides.extend(positive_peptides)
            binding_core_training_labels.extend([True] * n_pos)
            binding_core_training_weights.extend([1.0 / n_pos] * n_pos)
            binding_core_training_contigs.extend([g.contig] * n_pos)


            negative_peptides = []
            for i in range(len(g.contig) - k + 1):
                seq = g.contig[i:i + k]
                if seq not in positive_peptides:
                    negative_peptides.append(seq)
            n_neg = len(negative_peptides)
            if n_neg > 0:
                binding_core_training_peptides.extend(negative_peptides)
                binding_core_training_labels.extend([False] * n_neg)
                binding_core_training_weights.extend([1.0 / n_neg] * n_neg)
                binding_core_training_contigs.extend([g.contig] * n_neg)
        return Dataset(
            alleles=None,
            peptides=binding_core_training_peptides,
            labels=binding_core_training_labels,
            weights=binding_core_training_weights,
            contigs=binding_core_training_contigs)


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
            result[group_id] = self.select_best_peptide(peptides)
        return result

    def _restrict_dataset_to_best_binding_cores(self, dataset):
        positive_dataset = dataset[dataset.labels]
        negative_dataset = dataset[~dataset.labels]
        positive_peptide_groups = groupby(
            positive_dataset.peptides,
            keys=positive_dataset.contigs)
        contig_to_binding_core = self._find_binding_cores(positive_peptide_groups)
        n_kept = len(contig_to_binding_core)
        positive_subset = Dataset(
            alleles=dataset.alleles[:n_kept],
            peptides=list(contig_to_binding_core.values()),
            contigs=list(contig_to_binding_core.keys()))
        return negative_dataset.combine(positive_subset)

    def fit_predict(self, hit_peptides):
        """
        Given a set of MHCII ligands, assemble their sequences into longer
        contigs, infer candidate binding core regions, iteratively train
        a binding core classifier.
        Returns list of SequenceGroup objects (one for each contig) and
        dictionary mapping each contig to tuple of subsequences and scores
        for each subsequence.
        """
        sequence_groups = assemble_into_sequence_groups(hit_peptides)
        return self.fit_predict_sequence_groups(sequence_groups)

    def fit_dataset(self, dataset):
        pos_dataset = dataset[dataset.labels]
        sequence_groups = pos_dataset.to_sequence_groups()
        self.fit_sequence_groups(sequence_groups)
        return self

    def fit_predict_sequence_groups(self, sequence_groups):
        """
        Trains binding core predictor from list of SequenceGroup objects and
        then returns subsequence predictions for all kmers within sequence
        group contigs.
        """
        self.fit_sequence_groups(sequence_groups)
        contig_to_predictions = self.predict_subsequence_scores_for_sequence_groups(
            sequence_groups)
        return sequence_groups, contig_to_predictions

    def fit_sequence_groups(self, sequence_groups):
        """
        Iterative training of a fixed length binding core classifier which is
        then used to refine the training data.
        """
        dataset = self._build_training_set(sequence_groups)
        best_score = 0
        for i in range(self.max_iters):
            if self.model is None:
                # if a model hasn't been trained yet then just use the initial
                # training set where all candidate binding cores have equal weight
                current_dataset = dataset
            else:
                current_dataset = self._restrict_dataset_to_best_binding_cores(dataset)

            selected_binding_cores = current_dataset[current_dataset.labels].peptides
            if len(selected_binding_cores) == 0:
                raise ValueError("No binding cores in dataset: %s" % dataset)
            # pssm_score = compatibility_score_for_binding_cores(selected_binding_cores)
            self.model, auc, _ = self._cv_fit(current_dataset)
            if auc > best_score:
                best_score = auc
            else:
                return auc
        return best_score

    def encode_dataset(self, dataset):
        return self.encode_arrays(
            peptides=dataset.peptides,
            labels=dataset.labels,
            weights=dataset.weights,
            group_ids=dataset.group_ids)

    def encode_arrays(self, peptides, labels, weights, group_ids=None):
        X = self.encode_peptides(peptides)
        n_samples = len(X)
        y = np.array(labels)
        assert n_samples == len(y), len(y)
        weights = np.array(weights)
        if group_ids is None:
            group_ids = np.arange(n_samples)
        else:
            group_ids = np.array(group_ids)
        assert len(group_ids) == n_samples
        return X, y, weights, group_ids

    def encode_peptides(self, peptides, length=None):
        if length is None:
            length = self.length
        if any(len(p) != length for p in peptides):
            raise ValueError(
                "Expected all peptides to be %d in length, got %s" % (
                    length,
                    {len(p) for p in peptides}))
        X = self.encoder.encode_onehot(
            peptides,
            max_peptide_length=length)
        n_samples, expect_peptide_length, expect_n_amino_acids = X.shape
        if expect_peptide_length != length:
            raise ValueError("Expected 2nd dim to be %d but got %d" % (
                length, expect_peptide_length))
        if expect_n_amino_acids != 20:
            raise ValueError("Expected final dimension to be 20 but got %d" % (
                expect_n_amino_acids,))

        n_dims = length * 20
        X = X.reshape((n_samples, n_dims))
        assert len(X) == len(peptides), X.shape
        return X

    def select_best_peptide(self, peptides):
        """
        Returns peptide among the inputs with the highest predicted
        binding core score (along with the score itself)
        """
        scores = self.predict_scores_for_peptides(peptides)
        return peptides[np.argmax(scores)]

    def predict_labels_for_peptides(self, peptides):
        X = self.encode_peptides(peptides)
        assert self.model is not None
        return self.model.predict(X)

    def predict_scores_for_peptides(self, peptides):
        X = self.encode_peptides(peptides)
        assert self.model is not None
        return self.model.predict_proba(X)[:, -1]

    def _pad_short_peptide(self, peptide):
        """
        Given a k-mer where k < 9, turn it into a 9mer by padding it on
        the right and left by every possible amino acid.
        """
        peptides = []
        n_missing = self.length - len(peptide)
        for aa in self.encoder.tokens:
            # try padding sequences which are too short with all possible
            # amino acids
            padding = aa * n_missing
            peptides.append(padding + peptide)
            peptides.append(peptide + padding)
        return peptides

    def predict_binding_core(self, sequence):
        """
        Given a long sequence, return binding core within it
        """
        return self.predict_top_binding_cores(sequence, n=1)[0]

    def predict_top_binding_cores_with_indices(self, sequence, n=1):
        """
        Return up to `n` top scoring binding core sequence indices
        """

        subsequences, scores = self.predict_subsequence_scores(sequence)
        if n == 1:
            i = np.argmax(scores)
            return [subsequences[i]], [i]
        else:
            descending_indices = np.argsort(scores)[::-1]
            top_indices = descending_indices[:n]
            return [subsequences[i] for i in top_indices], top_indices

    def predict_top_binding_cores(self, sequence, n=1):
        """
        Return up to `n` top scoring binding core sequences
        """

        subsequences, scores = self.predict_subsequence_scores(sequence)
        if n == 1:
            return [subsequences[np.argmax(scores)]]
        else:
            descending_indices = np.argsort(scores)[::-1]
            top_indices = descending_indices[:n]
            return [subsequences[i] for i in top_indices]


    def _subsequences(self, sequence):
        k = self.length
        if len(sequence) >= k:
            return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

        else:
            return self._pad_short_peptide(sequence)

    def predict_subsequence_scores(self, sequence):
        """
        Returns tuple of (subsequences, scores)
        """
        subsequences = self._subsequences(sequence)
        scores = self.predict_scores_for_peptides(subsequences)
        if len(sequence) >= self.length:
            return (subsequences, scores)
        else:
            return [sequence], np.array([np.mean(scores)])

    def predict_subsequence_scores_dict(self, sequences):
        return {
            s: self.predict_subsequence_scores(s) for s in sequences
        }

    def predict_binding_cores_for_sequence_groups(self, sequence_groups):
        result = {}
        for group in sequence_groups:
            if len(group.contig) < self.length:
                result[group.contig] = group.contig
            else:
                binding_core_seq = self.predict_binding_core(group.contig)
                result[group.contig] = binding_core_seq
        return result

    def predict_subsequence_scores_for_sequence_groups(self, sequence_groups):
        result = {}
        for group in sequence_groups:
            result[group.contig] = self.predict_subsequence_scores(group.contig)
        return result

    def predict_peptides(self, long_sequences):
        """
        For each long sequence, predict score for every subsequence
        and then return mean of those scores.
        """
        max_scores = np.zeros(len(long_sequences), dtype="float64")
        for i, sequence in enumerate(long_sequences):
            subsequences = self._subsequences(sequence)
            scores = self.predict_scores_for_peptides(subsequences)
            if len(scores) == 1:
                max_scores[i] = scores[0]
            else:
                max_scores[i] = np.mean(scores)
        return max_scores
