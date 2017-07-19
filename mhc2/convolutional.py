import logging
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from pepnet import Predictor, SequenceInput, Output
import numpy as np

MAX_PEPTIDE_LENGTH = 30
FIRST_CONV_SIZES = {1: 4, 3: 4, 8: 4, 9: 16, 10: 4}
SECOND_CONV_SIZES = {1: 8, 3: 8, 5: 8, 7: 8, 9: 8}
CONV_DROPOUT = 0.25
CONV_ACTIVATION = "relu"
DENSE_DROPOUT = 0.25
DENSE_ACTIVATION = "relu"
GLOBAL_BATCH_NORMALIZATION = True
EMBEDDING_DIM = 24
BATCH_SIZE = 32

class ConvolutionalPredictor(object):
    def __init__(
            self,
            max_peptide_length=30,
            embedding_dim=EMBEDDING_DIM,
            conv_filter_sizes=[FIRST_CONV_SIZES, SECOND_CONV_SIZES],
            conv_dropout=CONV_DROPOUT,
            conv_activation="relu",
            global_pooling_batch_normalization=GLOBAL_BATCH_NORMALIZATION,
            dense_layer_sizes=[],
            dense_dropout=DENSE_DROPOUT,
            dense_activation=DENSE_ACTIVATION,
            add_start_tokens=True,
            add_stop_tokens=True,
            batch_size=BATCH_SIZE,
            n_models=3,
            training_patience=3,
            max_training_epochs=100):
        self.max_peptide_length = max_peptide_length
        self.embedding_dim = embedding_dim
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_dropout = conv_dropout
        self.conv_activation = conv_activation
        self.global_pooling_batch_normalization = global_pooling_batch_normalization
        self.dense_layer_sizes = dense_layer_sizes
        self.dense_dropout = dense_dropout
        self.dense_activation = dense_activation
        self.add_start_tokens = add_start_tokens
        self.add_stop_tokens = add_stop_tokens
        self.batch_size = batch_size
        self.n_models = n_models
        self.training_patience = training_patience
        self.max_training_epochs = max_training_epochs
        self.models = []
        self.model_weights = []

    def _tile_over_long_peptides(self, peptides):
        new_peptides = []
        new_indices = []
        new_counts = []

        k = self.max_peptide_length
        for i, p in enumerate(peptides):
            if len(p) <= k:
                new_peptides.append(p)
                new_indices.append(i)
                new_counts.append(1)
            else:
                 # tile over long peptides, generating multiple shorter
                # peptides
                half_len = self.max_peptide_length // 2
                count = 0
                for start in range(0, len(p), half_len):
                    new_peptides.append(p[start:start + k])
                    new_indices.append(i)
                    count += 1
                new_counts.extend([count] * count)

        return new_peptides, np.array(new_indices), np.array(new_counts)

    def _expand_arrays_for_long_peptides(self, peptides, labels, weights):
        """
        Since some peptides might be longer than the max sequence
        supported by this model, this helper expanded peptides
        by tiling over them and returns a 4 element tuple with the
        following entries:
            - peptides
            - labels
            - weights
            - indices mapping back to original sequences
        """
        new_peptides, new_indices, new_counts = \
            self._tile_over_long_peptides(peptides)
        assert len(new_peptides) == len(new_indices) == len(new_counts)
        new_labels = np.array([labels[i] for i in new_indices])
        new_weights = np.array([weights[i] for i in new_indices])
        new_weights /= new_counts
        assert len(new_labels) == len(new_weights) == len(new_peptides)
        assert len(new_peptides) >= len(peptides)
        return new_peptides, new_labels, new_weights, new_indices

    def _expand_dataset_for_long_peptides(self, dataset):
        return self._expand_arrays_for_long_peptides(
            peptides=dataset.peptides,
            labels=dataset.labels,
            weights=dataset.weights)

    def _make_model(self):
        return Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=self.max_peptide_length,
                add_start_tokens=self.add_start_tokens,
                add_stop_tokens=self.add_stop_tokens,
                variable_length=True,
                embedding_dim=self.embedding_dim,
                conv_filter_sizes=self.conv_filter_sizes,
                conv_activation=self.conv_activation,
                conv_dropout=self.conv_dropout,
                global_pooling=True,
                global_pooling_batch_normalization=self.global_pooling_batch_normalization),
            outputs=Output(
                1,
                activation="sigmoid",
                dense_layer_sizes=self.dense_layer_sizes,
                dense_activation=self.dense_activation,
                dense_dropout=self.dense_dropout))

    def _train(self, model, train_dataset, test_dataset):
        epochs_since_improvement = 0
        last_best_auc = 0
        last_best_model_params = None
        train_peptides, train_labels, train_weights, _ = \
            self._expand_dataset_for_long_peptides(train_dataset)
        test_peptides, test_labels, test_weights, _ = \
                self._expand_dataset_for_long_peptides(test_dataset)

        for epoch in range(self.max_training_epochs):

            model.fit(
                {"peptide": train_peptides},
                train_labels,
                sample_weight=train_weights,
                batch_size=self.batch_size,
                epochs=1)
            test_pred = model.predict({"peptide": test_peptides})
            test_auc = roc_auc_score(
                y_true=test_labels,
                y_score=test_pred,
                sample_weight=test_weights)
            if test_auc > last_best_auc:
                epochs_since_improvement = 0
                last_best_auc = test_auc
                last_best_model_params = model.get_weights()
            else:
                epochs_since_improvement += 1
            print("[Convolutional] -- epoch %d: test AUC = %0.4f%s" % (
                epoch + 1,
                test_auc,
                " (*)" if epochs_since_improvement == 0 else ""))
            if epochs_since_improvement >= self.training_patience:
                break
        if last_best_model_params is None:
            logging.warn("Best weights = None!")
        else:
            model.set_weights(last_best_model_params)
        rescaled_auc = max(0, (2 * (last_best_auc - 0.5))) ** 2
        return rescaled_auc

    def fit_dataset(self, dataset):
        assert len(dataset.unique_alleles()) == 1, \
            "Can only train one allele at a time"
        self.models = []
        dataset = dataset.shuffle()
        cv = GroupKFold(n_splits=self.n_models)
        for fold, (train_idx, test_idx) in enumerate(cv.split(
                X=dataset.peptides,
                y=dataset.labels,
                groups=dataset.group_ids)):
            print("[Convolutional] Training subset %d/%d" % (fold + 1, self.n_models))
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
            model = self._make_model()
            weight = self._train(model, train_dataset=train_dataset, test_dataset=test_dataset)
            self.models.append(model)
            self.model_weights.append(weight)

    def predict_peptides(self, peptides):
        assert len(self.models) > 0
        assert len(self.models) == len(self.model_weights)

        scores = np.zeros(len(peptides), dtype="float64")
        weights = np.zeros(len(peptides), dtype="float64")
        extended_peptides, extended_peptide_indices = self._tile_over_long_peptides(peptides)

        assert len(extended_peptides) == len(extended_peptide_indices)
        assert len(extended_peptides) >= len(peptides)
        for model, weight in zip(self.model, self.model_weights):
            extended_scores = model.predict(extended_peptides) * weight
            for i, x in zip(extended_peptide_indices, extended_scores):
                scores[i] += x
                weights[i] += weight
        return scores / weights
