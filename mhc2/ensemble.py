class Ensemble(object):
    """
    Ensemble of allele-specific Class II MHC binding predictors
    which can be serialized, trained, and used to make predictions.
    """
    pass


def make_predictor(
        conv_filter_sizes,
        conv_dropout=0.25,
        conv_activation="relu",
        global_pooling_batch_normalization=True,
        dense_layer_sizes=[],
        dense_dropout=0.25,
        dense_activation="relu"):
    return Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=MAX_LENGTH,
                add_start_tokens=True,
                add_stop_tokens=True,
                variable_length=True,
                embedding_dim=24,
                conv_filter_sizes=conv_filter_sizes,
                conv_activation=conv_activation,
                conv_dropout=conv_dropout,
                global_pooling=True,
                global_pooling_batch_normalization=global_pooling_batch_normalization),
            outputs=Output(
                1,
                activation="sigmoid",
                dense_layer_sizes=dense_layer_sizes,
                dense_activation="relu",
                dense_dropout=dense_dropout))

def train_predictors_for_allele(
        dataset,
        n_cv_splits=N_CV_SPLITS,
        max_epochs=MAX_EPOCHS,
        csv_writer=None):
    outer_cv = GroupKFold(n_splits=n_cv_splits)
    inner_cv = GroupKFold(n_splits=2)
    allele = list(set(dataset.alleles))[0]

    for outer_fold_idx, (train_idx, val_and_test_idx) in enumerate(outer_cv.split(
            X=dataset.peptides,
            y=dataset.labels,
            groups=dataset.group_ids)):
        train_dataset = dataset[train_idx]
        n_train = len(train_dataset)
        n_train_pos = train_dataset.labels.sum()
        val_and_test_dataset = dataset[val_and_test_idx]
        for inner_fold_idx, (val_idx, test_idx) in enumerate(inner_cv.split(
                X=val_and_test_dataset.peptides,
                y=val_and_test_dataset.labels,
                groups=val_and_test_dataset.group_ids)):

            val_dataset = val_and_test_dataset[val_idx]
            n_val = len(val_dataset)
            n_val_pos = val_dataset.labels.sum()
            test_dataset = val_and_test_dataset[test_idx]
            n_test = len(test_dataset)
            n_test_pos = test_dataset.labels.sum()

            print(
                "%s: fold %d/%d, val fold %d/2, # training pos = %d/%d, # val pos = %d/%d" % (
                    allele,
                    outer_fold_idx + 1,
                    n_cv_splits,
                    inner_fold_idx + 1,
                    n_train_pos,
                    n_train,
                    n_val_pos
                    n_val))
            if csv_writer:
                csv_row = {
                    "allele": allele,
                    "outer_fold": outer_fold_idx,
                    "inner_fold": inner_fold_idx,
                    "n_train": n_train,
                    "n_train_pos": n_train_pos,
                    "n_val": n_val,
                    "n_val_pos": n_val_pos,
                    "n_test": n_test,
                    "n_test_pos": n_test_pos,
                }
            for i, hyperparameters in enumerate(all_hyperparameters):
                if csv_writer:
                    for k, v in hyperparameters.items():
                        def to_string(v):
                            if isinstance(v, dict):
                                v = ";".join([
                                    "%s:%s" % (width, n_filters)
                                    for (width, n_filters) in v.items()])
                            elif isinstance(v, (list, tuple)):
                                v = ";".join([to_string(vi) for vi in v])
                            else:
                                return str(v)
                        csv_row[k] = v
                    print("==> %s" % (csv_row,))
                else:
                    print("==> %s" % (hyperparameters,))
                predictor = make_predictor(**hyperparameters)
                predictor.save_diagram("%s_%d.png" % (allele, i))

                epochs_since_improvement = 0
                last_best_auc = 0
                last_best_weights = None
                for epoch in range(MAX_EPOCHS):
                    predictor.fit(
                        {"peptide": train_dataset.peptides},
                        train_dataset.labels,
                        sample_weight=train_dataset.weights,
                        batch_size=64,
                        epochs=1,
                        validation_data=(
                            val_dataset.peptides,
                            val_dataset.labels,
                            val_dataset.weights))
                    val_pred = predictor.predict({"peptide": val_dataset.peptides})
                    val_auc = roc_auc_score(
                        y_true=val_dataset.labels,
                        y_score=val_pred,
                        sample_weight=val_dataset.weights)
                    if val_auc > last_best_auc:
                        epochs_since_improvement = 0
                        last_best_auc = val_auc
                        last_best_weights = predictor.get_weights()
                    else:
                        epochs_since_improvement += 1
                    print("-- epoch %d: val AUC = %0.4f%s" % (
                        epoch + 1,
                        val_auc,
                        " (*)" if epochs_since_improvement == 0 else ""))
                    if epochs_since_improvement >= PATIENCE:
                        break
                if last_best_weights is None:
                    logging.warn("Best weights = None!")
                else:
                    predictor.set_weights(last_best_weights)
                test_pred = predictor.predict({"peptide": test_dataset.peptides})
                test_auc = roc_auc_score(
                    y_true=test_dataset.labels,
                    y_score=test_pred,
                    sample_weight=test_dataset.weights)
                if csv_writer:
                    csv_row["auc"] = test_auc
                    csv_writer.writerow(csv_row)
                print(">>>\n-- Test AUC = %0.4f" % test_auc)
                weight = max(0, (2 * (test_auc - 0.5))) ** 2
                yield predictor, hyperparameters, weight

def train_predictors(
        dataset,
        n_cv_splits=N_CV_SPLITS,
        max_epochs=MAX_EPOCHS,
        csv_log_filename="allele_specific_log.csv"):
    """
    Generate which returns tuples with these fields:
        (allele, model, hyperparameters, weight)
    """
    hyperparameter_keys = sorted(all_hyperparameters[0].keys())
    with open(csv_log_filename, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=[
            "allele",
            "outer_fold",
            "inner_fold",
            "auc",
            "n_train",
            "n_train_pos",
            "n_val",
            "n_val_pos",
            "n_test",
            "n_test_pos"] + hyperparameter_keys)
        csv_writer.writeheader()
        for allele, allele_dataset in dataset.groupby_allele():
            for (model, hyperparameters, weight) in train_predictors_for_allele(
                    allele_dataset,
                    n_cv_splits=n_cv_splits,
                    max_epochs=max_epochs,
                    csv_writer=csv_writer):
                f.flush()
                yield allele, model, hyperparameters, weight

def load_extra_data(hits_file="new_hits.txt", decoys_file="new_decoys.txt"):
    dataframes = []
    for filename, label in [(hits_file, True), (decoys_file, False)]:
        with open(filename) as f:
            peptides = [line.strip().upper() for line in f]
            print("Loaded %d peptides with hit=%s" % (
                len(peptides),
                label))
            df = pd.DataFrame()
            df["peptide"] = peptides
            df["hit"] = [label] * len(peptides)
            print("-- length distribution:")
            print(df.peptide.str.len().value_counts())
            dataframes.append(df)
    combined_df = pd.concat(dataframes)
    hit_peptide_set = set(combined_df.peptide[combined_df.hit])
    decoy_peptide_set = set(combined_df.peptide[~combined_df.hit])
    print("-- %d peptides with both hit labels" % (
        len(hit_peptide_set.intersection(decoy_peptide_set))))
    bad_aa_mask = combined_df.peptide.str.contains("X")
    print("-- dropping %d peptides with 'X'" % bad_aa_mask.sum())
    combined_df = combined_df[~bad_aa_mask]
    return combined_df

if __name__ == "__main__":
    dataset = load_mass_spec_hits_and_decoys_grouped_by_nested_sets(
        DECOY_FACTOR,
        max_decoy_length=MAX_LENGTH)
    hit_weight = dataset.weights[dataset.labels].sum()
    decoy_weight = dataset.weights[~dataset.labels].sum()
    if not np.allclose(hit_weight, decoy_weight):
        logging.warn(
            "%0.4f weight for hits, %0.4f for decoys" % (
                hit_weight, decoy_weight))
    extra_peptides_df = load_extra_data()

    pred_peptides = list(extra_peptides_df.peptide)
    allele_to_predictions = defaultdict(
        lambda: np.zeros(len(pred_peptides), dtype="float64"))
    allele_to_sum_weights = defaultdict(
        lambda: 0.0)
    for i, (allele, model, hyperparameters, weight) in enumerate(
            train_predictors(dataset)):
        print("==> PRED: %s #%d %s (weight=%0.4f)" % (
            allele, i + 1, hyperparameters, weight))
        pred = model.predict(pred_peptides)
        allele_to_predictions[allele] += pred * weight
        allele_to_sum_weights[allele] += weight
        for allele, sum_pred in allele_to_predictions.items():
            extra_peptides_df[allele] = sum_pred / allele_to_sum_weights[allele]
        extra_peptides_df.to_csv("ensemble_predictions.csv")
