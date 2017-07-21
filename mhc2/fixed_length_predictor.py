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

import numpy as np
from pepnet import Predictor, NumericInput, Output

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from .binding_core_predictor import BindingCorePredictor
from .common import groupby, groupby_average_array, groupby_array_fn
from .assembly import assemble_into_sequence_groups
from .decoys import generate_decoy_sequence_groups
from .dataset import Dataset

amino_acids = sorted(list("GALMFWKQESPVICYHRNDT"))
assert len(amino_acids) == 20

class FixedLengthPredictor(object):
    def __init__(
            self,
            binding_core_size=9,
            n_binding_cores_train=1,
            n_binding_cores_predict=1,
            n_nterm=1,
            n_cterm=1,
            max_binding_core_iters=100,
            epochs=20,
            n_models=2,
            lr_penalty="l2",
            lr_solver="liblinear",
            combine_predictions_fn=np.max):
        self.binding_core_size = binding_core_size
        self.binding_core_predictors = []
        self.n_model = n_models
        self.models = []
        self.binding_core_predictors = []
        self.n_binding_cores_train = n_binding_cores_train
        self.n_binding_cores_predict = n_binding_cores_predict
        self.n_nterm = n_nterm
        self.n_cterm = n_cterm
        self.epochs = epochs
        self.max_binding_core_iters = max_binding_core_iters
        self.n_models = n_models
        self.lr_penalty =lr_penalty
        self.lr_solver = lr_solver
        self.combine_predictions_fn = combine_predictions_fn

    def _make_binding_core_predictor(self, max_binding_core_iters=None):
        if not max_binding_core_iters:
            max_binding_core_iters = self.max_binding_core_iters
        return BindingCorePredictor(
            length=self.binding_core_size,
            max_iters=max_binding_core_iters)

    def _make_nn_model(
            self,
            n_features,
            dense_layer_sizes=[20],
            batch_normalization=True):
        return Predictor(
                inputs=NumericInput(
                    n_features,
                    dense_layer_sizes=dense_layer_sizes,
                    dense_batch_normalization=batch_normalization),
                outputs=Output(1, activation="sigmoid"))

    def _make_lr_model(self):
        return LogisticRegressionCV(
            penalty=self.lr_penalty,
            solver=self.lr_solver)

    def _encode(
            self,
            binding_core_predictor,
            sequences,
            n_binding_cores=1,
            sequence_labels=True,
            group_ids=None):
        if group_ids is None:
            group_ids = np.arange(len(sequences))
        if isinstance(sequence_labels, (int, bool)):
            sequence_labels = np.array([sequence_labels] * len(sequences))
        binding_core_list = []
        distance_from_Nterm = []
        distance_from_Cterm = []
        total_lengths = []
        Nterm_list = []
        Cterm_list = []

        left_amino_acid_counts = {
            aa: []
            for aa in amino_acids
        }
        right_amino_acid_counts = {
            aa: []
            for aa in amino_acids
        }
        total_cysteine_count = []
        labels = []
        weights = []
        kept_group_id_list = []
        k = self.binding_core_size
        sequence_to_label_dict = {
            sequence: label
            for (sequence, label)
            in zip(sequences, sequence_labels)

        }
        # pre-create all possible padding sizes necessary for peptides
        # smaller than the binding core size
        padding_list = ["A" * i for i in range(k)]
        for group_id, peptides in groupby(sequences, group_ids).items():
            count = 0
            for c in peptides:

                label = sequence_to_label_dict[c]
                length = len(c)
                if length < k:
                    # pad short peptides with alanines to still be able to make
                    # a prediction
                    n_padding = k - length
                    padding = padding_list[n_padding]
                    c = c + padding
                    assert len(c) == k
                    curr_binding_cores = [c] * n_binding_cores
                    indices = [0] * n_binding_cores
                else:
                    curr_binding_cores, indices = \
                        binding_core_predictor.predict_top_binding_cores_with_indices(
                            sequence=c,
                            n=n_binding_cores)
                Nterm = c[:self.n_nterm]
                Cterm = c[-self.n_cterm:]
                for binding_core, idx in zip(curr_binding_cores, indices):
                    binding_core_list.append(binding_core)
                    labels.append(label)
                    distance_from_Nterm.append(idx)
                    curr_distance_from_Cterm = (
                        len(c) - k - idx)
                    distance_from_Cterm.append(curr_distance_from_Cterm)
                    Nterm_list.append(Nterm)
                    Cterm_list.append(Cterm)
                    kept_group_id_list.append(group_id)
                    total_lengths.append(length)
                    for aa in amino_acids:
                        left_amino_acid_counts[aa].append(sum(ci == aa for ci in c[:idx]))
                        right_amino_acid_counts[aa].append(sum(ci == aa for ci in c[idx + k:]))
                    total_cysteine_count.append(
                        sum(ci == "C" for ci in c))
                    count += 1
            if count == 0:
                continue
            weight = 1.0 / count
            weights.extend([weight] * count)
        n_samples = len(binding_core_list)
        if len(distance_from_Nterm) != n_samples:
            raise ValueError(
                "Expected %d samples but got %d distance_from_Nterm" % (
                    n_samples, len(distance_from_Nterm)))
        if len(distance_from_Cterm) != n_samples:
            raise ValueError(
                "Expected %d samples but got %d distance_from_Cterm" % (
                    n_samples, len(distance_from_Cterm)))
        if len(Nterm_list) != n_samples:
            raise ValueError(
                "Expected %d samples but got %d Nterm_list" % (
                    n_samples, len(Nterm_list)))
        if len(Cterm_list) != n_samples:
            raise ValueError(
                "Expected %d samples but got %d Cterm_list" % (
                    n_samples, len(Cterm_list)))
        if len(labels) != n_samples:
            raise ValueError(
                "Expected %d samples but got %d labels" % (
                    n_samples, len(labels)))
        if len(weights) != n_samples:
            raise ValueError(
                "Expected %d samples but got %d weights" % (
                    n_samples, len(weights)))
        if len(total_lengths) != n_samples:
            raise ValueError(
                "Expected %d samples but got %d lengths" % (
                    n_samples, len(total_lengths)))
        sequence_features = [
            bc + nterm + cterm
            for (bc, nterm, cterm)
            in zip(binding_core_list, Nterm_list, Cterm_list)]
        assert len(sequence_features) == n_samples

        numeric_distance_features = np.array([
            distance_from_Nterm, distance_from_Cterm]).T
        assert len(numeric_distance_features) == n_samples
        X_dist = np.hstack([
            numeric_distance_features == 0,
            numeric_distance_features == 1,
            numeric_distance_features == 2,
            numeric_distance_features == 3,
            numeric_distance_features > 3,
        ])
        assert len(X_dist) == n_samples
        total_lengths = np.array(total_lengths)
        X_length = np.array([
            total_lengths < 12,
            total_lengths > 20,
            total_lengths > 35,
        ]).T
        assert len(X_length) == n_samples, X_length.shape
        assert len(X_length.shape) == 2, X_length.shape
        X_flanking_aa_counts = []
        for aa in amino_acids:
            X_flanking_aa_counts.append(np.array(left_amino_acid_counts[aa]))
            X_flanking_aa_counts.append(np.array(right_amino_acid_counts[aa]))
        X_flanking_aa_counts.append(np.array(total_cysteine_count))
        X_flanking_aa_counts = np.array(X_flanking_aa_counts).T
        assert len(X_flanking_aa_counts) == n_samples
        assert X_flanking_aa_counts.shape[1] == 40 + 1 # 1 extra for cysteines
        assert len(X_flanking_aa_counts.shape) == 2, X_flanking_aa_counts.shape
        X_numeric = np.hstack([
            X_dist,
            X_length,
            X_flanking_aa_counts,
        ])
        assert X_numeric.shape[0] == n_samples, X_numeric.shape
        assert X_numeric.shape[1] == (
            X_dist.shape[1] + X_length.shape[1] + X_flanking_aa_counts.shape[1])

        X_aa = binding_core_predictor.encode_peptides(
            sequence_features,
            length=self.binding_core_size + self.n_nterm + self.n_cterm)
        assert len(X_aa) == n_samples
        X = np.hstack([X_numeric, X_aa])
        assert len(X) == n_samples
        weights = np.array(weights)
        assert len(weights) == n_samples
        labels = np.array(labels)
        assert len(labels) == n_samples
        group_ids = np.array(kept_group_id_list)
        assert len(group_ids) == n_samples
        print("-- Generated feature matrix with shape %s" % (X.shape,))
        return X, labels, weights, group_ids

    def _encode_dataset(
                self,
                binding_core_predictor,
                dataset,
                n_binding_cores=1,
                label=None):
        if label is None:
            sequence_labels = dataset.labels
        else:
            sequence_labels = label
        return self._encode(
            binding_core_predictor=binding_core_predictor,
            sequences=dataset.peptides,
            sequence_labels=sequence_labels,
            group_ids=dataset.group_ids,
            n_binding_cores=n_binding_cores)

    def _encode_sequence_groups(
            self,
            binding_core_predictor,
            sequence_groups,
            n_binding_cores=1,
            label=None):
        dataset = Dataset.from_sequence_groups(
            sequence_groups,
            label=label)
        return self._encode_dataset(
            binding_core_predictor=binding_core_predictor,
            dataset=dataset,
            n_binding_cores=n_binding_cores,
            label=label)

    def _create_training_data(
            self,
            binding_core_predictor,
            hit_sequence_groups,
            decoy_sequence_groups,
            balance_class_weights=True):
        X_pos, y_pos, weights_pos, group_ids_pos = \
            self._encode_sequence_groups(
                binding_core_predictor=binding_core_predictor,
                sequence_groups=hit_sequence_groups,
                label=True,
                n_binding_cores=self.n_binding_cores_train)
        X_neg, y_neg, weights_neg, group_ids_neg = \
            self._encode_sequence_groups(
                binding_core_predictor=binding_core_predictor,
                sequence_groups=decoy_sequence_groups,
                label=False,
                n_binding_cores=self.n_binding_cores_train)
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])
        if balance_class_weights:
            weights_pos_total = weights_pos.sum()
            weights_neg_total = weights_neg.sum()
            weights_ratio = weights_pos_total / weights_neg_total
        else:
            weights_ratio = 1.0
        weights = np.concatenate([weights_pos, weights_neg * weights_ratio])
        group_ids = np.concatenate([
            group_ids_pos,
            1 + group_ids_pos.max() + group_ids_neg])
        return X, y, weights, group_ids

    def fit_sequence_groups(
            self,
            hit_sequence_groups,
            decoy_sequence_groups):
        """
        Split hits in half to use one part to train the binding core predictor
        and the other to train the complete model.
        """
        self.models = []
        self.binding_core_predictors = []
        cv = KFold(self.n_models, shuffle=True)
        for fold, (idx1, idx2) in enumerate(cv.split(hit_sequence_groups)):
            print("[FixedLengthPredictor] Fitting model %d/%d" % (
                fold + 1, self.n_models))
            binding_core_groups = [hit_sequence_groups[i] for i in idx1]
            full_model_groups = [hit_sequence_groups[i] for i in idx2]
            binding_core_predictor = self._make_binding_core_predictor()
            binding_core_predictor.fit_sequence_groups(binding_core_groups)
            self.binding_core_predictors.append(binding_core_predictor)
            X, y, weights, _ = self._create_training_data(
                binding_core_predictor=binding_core_predictor,
                hit_sequence_groups=full_model_groups,
                decoy_sequence_groups=decoy_sequence_groups)
            full_model = self._make_lr_model()
            full_model.fit(X, y, sample_weight=weights)
            self.models.append(full_model)
        return self

    def fit_dataset(self, dataset):
        pos_dataset = dataset[dataset.labels]
        neg_dataset = dataset[~dataset.labels]
        pos_sequence_groups = pos_dataset.to_sequence_groups()
        neg_sequence_groups = neg_dataset.to_sequence_groups()
        return self.fit_sequence_groups(pos_sequence_groups, neg_sequence_groups)

    def _predict_model_scores(self, X, model):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, -1]
        else:
            assert hasattr(model, 'predict_scores')
            return model.predict_scores(X)

    def _predict_scores(self, X):
        """
        Predict scores for encoded array
        """
        scores = np.zeros(len(X), dtype="float64")
        for model in self.models:
           scores += self._predict_model_scores(X, model)
        return scores / self.n_models

    def fit_positive_sequence_groups(self, hit_sequence_groups, decoys_per_hit=10):
        n_hit_groups = len(hit_sequence_groups)
        n_decoy_loci = n_hit_groups * decoys_per_hit
        decoy_sequence_groups = generate_decoy_sequence_groups(
            n_decoy_loci=n_decoy_loci)
        self.fit_sequence_groups(
            hit_sequence_groups=hit_sequence_groups,
            decoy_sequence_groups=decoy_sequence_groups)
        return self

    def fit_peptides(self, peptides, decoys_per_hit=10):
        hit_sequence_groups = assemble_into_sequence_groups(peptides)
        self.fit_positive_sequence_groups(
            hit_sequence_groups,
            decoys_per_hit=decoys_per_hit)
        return self

    def fit_predict_peptides(self, peptides, decoys_per_hit=10):
        self.fit_peptides(peptides, decoys_per_hit=decoys_per_hit)
        return self.predict(peptides)

    def predict(self, sequences):
        n = len(sequences)
        y_pred_sum = np.zeros(n)
        combine_fn = self.combine_predictions_fn
        for binding_core_predictor, model in zip(
                self.binding_core_predictors, self.models):
            X, _, weights, original_indices = self._encode(
                binding_core_predictor=binding_core_predictor,
                sequences=sequences,
                n_binding_cores=self.n_binding_cores_predict)
            y_pred_multiple_binding_cores = self._predict_model_scores(X, model)
            # group predictions from multiple binding cores back to
            # their originating sequence
            y_pred_grouped = groupby_array_fn(
                y_pred_multiple_binding_cores,
                indices=original_indices,
                size=n,
                fn=combine_fn)
            y_pred_sum += y_pred_grouped
        y_pred_avg = y_pred_sum / self.n_models
        return y_pred_avg

    def predict_peptides(self, peptides):
        """
        Alias for FixedLengthPredictor.predict
        """
        return self.predict(peptides)

    def predict_sequences_with_group_ids(self, sequences, group_ids, n_groups):
        y_pred = self.predict(sequences)
        return groupby_average_array(
            y_pred,
            indices=group_ids,
            size=n_groups)

    def predict_dataset(self, dataset):
        return self.predict_sequences_with_group_ids(
            dataset.peptides,
            dataset.group_ids,
            n_groups=len(set(dataset.group_ids)))

    def predict_sequence_groups(self, sequence_groups):
        dataset = Dataset.from_sequence_groups(sequence_groups)
        return self.predict_dataset(dataset)
