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

from sklearn.linear_model import LogisticRegressionCV

from .binding_core_predictor import BindingCorePredictor
from .common import groupby
from .assembly import assemble_and_assign_to_sequence_groups
from .decoys import generate_decoy_sequence_groups

class FixedLengthPredictor(object):
    def __init__(
            self,
            binding_core_size=9,
            n_binding_cores_train=1,
            n_binding_cores_predict=1,
            n_nterm=1,
            n_cterm=1):
        self.binding_core_size = binding_core_size
        self.binding_core_predictor = self._make_binding_core_predictor()
        self.model = self._make_model()
        self.n_binding_cores_train = n_binding_cores_train
        self.n_binding_cores_predict = n_binding_cores_predict
        self.n_nterm = n_nterm
        self.n_cterm = n_cterm

    def _make_binding_core_predictor(self):
        return BindingCorePredictor(length=self.binding_core_size)

    def _make_model(self):
        return LogisticRegressionCV()

    def _flatten_sequence_groups(self, sequence_groups):
        child_sequences = []
        group_ids = []
        for group_id, group in enumerate(sequence_groups):
            for c in group.children:
                child_sequences.append(c)
                group_ids.append(group_id)
        return child_sequences, group_ids

    def _encode(self, sequences, label=True, group_ids=None):
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
        for group_id, peptides in groupby(sequences, group_ids).items():
            count = 0
            for c in peptides:
                if len(c) < self.binding_core_size:
                    continue
                curr_binding_cores = \
                    self.binding_core_predictor.predict_top_binding_cores(
                        sequence=c,
                        n=self.n_binding_cores_train)
                binding_core_list.extend(curr_binding_cores)
                Nterm = c[:self.n_nterm]
                Cterm = c[-self.n_cterm:]
                for binding_core in curr_binding_cores:
                    idx = c.index(binding_core)
                    distance_from_Nterm.append(idx)
                    distance_from_Cterm.append(
                        len(c) - self.binding_core_size - idx)
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
            label=True):
        sequences, group_ids = self._flatten_sequence_groups(sequence_groups)
        return self._encode(sequences=sequences, label=label, group_ids=group_ids)

    def create_training_data(
            self,
            hit_sequence_groups,
            decoy_sequence_groups,
            balance_class_weights=True):
        X_pos, y_pos, weights_pos, group_ids_pos = \
            self._encode_sequence_groups(hit_sequence_groups, label=True)
        X_neg, y_neg, weights_neg, group_ids_neg = \
            self._encode_sequence_groups(decoy_sequence_groups, label=False)
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
        self.model.fit(X, y, sample_weight=weights)

    def fit_predict_hits(self, hits, decoys_per_hit=10):
        hit_sequence_groups = assemble_and_assign_to_sequence_groups(hits)
        n_hit_groups = len(hit_sequence_groups)
        n_decoy_loci = n_hit_groups * decoys_per_hit
        decoy_sequence_groups = generate_decoy_sequence_groups(
            n_decoy_loci=n_decoy_loci)
        self.fit_sequence_groups(
            hit_sequence_groups=hit_sequence_groups,
            decoy_sequence_groups=decoy_sequence_groups)
        return self.predict(hits)

    def predict(self, sequences):
        n = len(sequences)
        X, _, weights, original_indices = self._encode(sequences)
        y_pred = self.model.predict_proba(X)[:, -1]
        idx_to_predictions = groupby(y_pred, original_indices)
        result = np.zeros(n)
        for idx, predictions in idx_to_predictions.items():
            result[idx] = np.mean(predictions)
        return result
