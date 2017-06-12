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

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

def train_predictors_for_allele(
        dataset,
        make_predictor_fn,
        n_cv_splits,
        max_epochs,
        patience,
        batch_size=64,
        csv_writer=None):
    outer_cv = GroupKFold(n_splits=n_cv_splits)
    inner_cv = GroupKFold(n_splits=2)
    allele = list(set(dataset.alleles))[0]

    for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(
            X=dataset.peptides,
            y=dataset.labels,
            groups=dataset.group_ids)):
        train_dataset = dataset[train_idx]
        n_train = len(train_dataset)
        n_train_pos = train_dataset.labels.sum()
        test_dataset = dataset[test_idx]
        n_test = len(test_dataset)
        n_test_pos = test_dataset.labels.sum()

        if csv_writer:
            csv_row = {
                "allele": allele,
                "outer_fold": outer_fold_idx,
                "inner_fold": inner_fold_idx,
                "n_train": n_train,
                "n_train_pos": n_train_pos,
                "n_test": n_test,
                "n_test_pos": n_test_pos,
            }

        predictor = make_predictor_fn()

        epochs_since_improvement = 0
        last_best_auc = 0
        last_best_weights = None
        for epoch in range(max_epochs):
            predictor.fit(
                {"peptide": train_dataset.peptides},
                train_dataset.labels,
                sample_weight=train_dataset.weights,
                batch_size=batch_size,
                epochs=1)
            test_pred = predictor.predict({"peptide": test_dataset.peptides})
            test_auc = roc_auc_score(
                y_true=test_dataset.labels,
                y_score=test_pred,
                sample_weight=test_dataset.weights)
            if test_auc > last_best_auc:
                epochs_since_improvement = 0
                last_best_auc = test_auc
                last_best_weights = predictor.get_weights()
            else:
                epochs_since_improvement += 1
            print("-- epoch %d: test AUC = %0.4f%s" % (
                epoch + 1,
                test_auc,
                " (*)" if epochs_since_improvement == 0 else ""))
            if epochs_since_improvement >= PATIENCE:
                break
        if last_best_weights is None:
            logging.warn("Best weights = None!")
        else:
            predictor.set_weights(last_best_weights)
        if csv_writer:
            csv_row["auc"] = last_best_auc
            csv_writer.writerow(csv_row)
        weight = max(0, (2 * (last_best_auc - 0.5))) ** 2
        yield predictor, weight

def train_allele_to_ensemble_dict(
        dataset,
        make_predictor_fn,
        n_cv_splits,
        max_epochs,
        patience,
        batch_size=64,
        csv_log_filename=None):
    """
    Generate which returns tuples with these fields:
        (allele, model, weight)
    """
    allele_to_ensemble_dict = defaultdict(Ensemble)
    if csv_log_filename:
        f = open(csv_log_filename, "w")
        csv_writer = csv.DictWriter(f, fieldnames=[
            "allele",
            "outer_fold",
            "inner_fold",
            "auc",
            "n_train",
            "n_train_pos",
            "n_test",
            "n_test_pos"])
        csv_writer.writeheader()
    else:
        csv_writer = None
        f = None

    for allele, allele_dataset in dataset.groupby_allele():
        for (model, weight) in train_predictors_for_allele(
                    dataset=allele_dataset,
                    make_predictor_fn=make_predictor_fn,
                    n_cv_splits=n_cv_splits,
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    csv_writer=csv_writer):
            if f:
                f.flush()
            allele_to_ensemble_dict[allele].add_model(model, weight)
    return allele_to_ensemble_dict

def train_model_collection(
        path,
        dataset,
        make_predictor_fn,
        n_cv_splits,
        max_epochs,
        patience,
        batch_size=64,
        csv_log_filename=None):
    model_collection = ModelCollection(path)
    allele_to_ensemble_dict = train_allele_to_ensemble_dict(
        dataset=dataset,
        make_predictor_fn=make_predictor_fn,
        n_cv_splits=n_cv_splits,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=64,
        csv_log_filename=None)
    for allele, ensemble in allele_to_ensemble_dict.items():
        model_collection.add_ensemble(allele, ensemble)
    return model_collection

