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
from .binding_core_predictor import BindingCorePredictor
from .common import groupby, groupby_average_array, groupby_max_array
from .assembly import assemble_into_sequence_groups
from .decoys import generate_decoy_sequence_groups
from .sequence_group import flatten_sequence_groups

class FixedLengthPredictor(object):
    def __init__(
            self,
            binding_core_size=9,
            n_binding_cores_train=1,
            n_binding_cores_predict=1,
            n_nterm=1,
            n_cterm=1,
            max_binding_core_iters=100,
            epochs=20):
        self.binding_core_size = binding_core_size
        self.binding_core_predictor = self._make_binding_core_predictor(
            max_binding_core_iters)
        self.model = None
        self.n_binding_cores_train = n_binding_cores_train
        self.n_binding_cores_predict = n_binding_cores_predict
        self.n_nterm = n_nterm
        self.n_cterm = n_cterm
        self.epochs = epochs

    def _make_binding_core_predictor(self, max_binding_core_iters=100):
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

    def _make_lr_model(self, penalty="l2"):
        return LogisticRegressionCV(penalty=penalty)

    def _encode(self, sequences, n_binding_cores=1, label=True, group_ids=None):
        if group_ids is None:
            group_ids = np.arange(len(sequences))
        binding_core_list = []
        distance_from_Nterm = []
        distance_from_Cterm = []
        Nterm_list = []
        Cterm_list = []
        labels = []
        weights = []
        kept_group_id_list = []
        k = self.binding_core_size
        for group_id, peptides in groupby(sequences, group_ids).items():
            count = 0
            for c in peptides:
                if len(c) < k:
                    continue
                curr_binding_cores, indices = \
                    self.binding_core_predictor.predict_top_binding_cores_with_indices(
                        sequence=c,
                        n=n_binding_cores)
                binding_core_list.extend(curr_binding_cores)
                Nterm = c[:self.n_nterm]
                Cterm = c[-self.n_cterm:]
                for binding_core, idx in zip(curr_binding_cores, indices):
                    distance_from_Nterm.append(idx)
                    curr_distance_from_Cterm = (
                        len(c) - self.binding_core_size - idx)
                    distance_from_Cterm.append(curr_distance_from_Cterm)
                    Nterm_list.append(Nterm)
                    Cterm_list.append(Cterm)
                    kept_group_id_list.append(group_id)
                    count += 1
            if count == 0:
                continue
            weight = 1.0 / count
            weights.extend([weight] * count)
            labels.extend([label] * count)
        n_samples = len(binding_core_list)
        assert len(distance_from_Nterm) == n_samples
        assert len(distance_from_Cterm) == n_samples
        assert len(Nterm_list) == n_samples
        assert len(Cterm_list) == n_samples
        assert len(labels) == n_samples
        assert len(weights) == n_samples
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
        X_aa =self.binding_core_predictor.encode_peptides(
            sequence_features,
            length=self.binding_core_size + self.n_nterm + self.n_cterm)
        assert len(X_aa) == n_samples
        X = np.hstack([X_dist, X_aa])
        assert len(X) == n_samples
        weights = np.array(weights)
        assert len(weights) == n_samples
        labels = np.array(labels)
        assert len(labels) == n_samples
        group_ids = np.array(kept_group_id_list)
        assert len(group_ids) == n_samples
        return X, labels, weights, group_ids

    def _encode_sequence_groups(
            self,
            sequence_groups,
            n_binding_cores=1,
            label=True):
        sequences, group_ids = flatten_sequence_groups(sequence_groups)
        return self._encode(
            sequences=sequences,
            label=label,
            group_ids=group_ids,
            n_binding_cores=n_binding_cores)

    def create_training_data(
            self,
            hit_sequence_groups,
            decoy_sequence_groups,
            balance_class_weights=True):
        X_pos, y_pos, weights_pos, group_ids_pos = \
            self._encode_sequence_groups(
                hit_sequence_groups,
                label=True,
                n_binding_cores=self.n_binding_cores_train)
        X_neg, y_neg, weights_neg, group_ids_neg = \
            self._encode_sequence_groups(
                decoy_sequence_groups,
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

    def fit_sequence_groups(self, hit_sequence_groups, decoy_sequence_groups):
        self.binding_core_predictor.fit_sequence_groups(hit_sequence_groups)
        X, y, weights, _ = self.create_training_data(
            hit_sequence_groups, decoy_sequence_groups)
        # self.model = self._make_model(n_features=X.shape[1])
        # self.model.fit(X, y, sample_weight=weights, epochs=self.epochs)
        self.model = self._make_lr_model()
        self.model.fit(X, y, sample_weight=weights)
        return self

    def _predict_scores(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, -1]
        else:
            assert hasattr(self.model, 'predict_scores')
            return self.model.predict_scores(X)

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
        X, _, weights, original_indices = self._encode(
            sequences,
            n_binding_cores=self.n_binding_cores_predict)
        y_pred = self._predict_scores(X)
        return groupby_max_array(
            y_pred,
            indices=original_indices,
            size=len(sequences))

    def predict_sequences_with_group_ids(self, sequences, group_ids, n_groups):
        y_pred = self.predict(sequences)
        return groupby_average_array(
            y_pred,
            indices=group_ids,
            size=n_groups)

    def predict_sequence_groups(self, sequence_groups):
        sequences, group_ids = flatten_sequence_groups(sequence_groups)
        return self.predict_sequences_with_group_ids(
            sequences,
            group_ids,
            n_groups=len(sequence_groups))
