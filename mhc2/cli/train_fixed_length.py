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

import sys
from argparse import ArgumentParser
from collections import defaultdict

from Bio import motifs
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq

from pepnet import Encoder
import numpy as np
from sklearn.linear_model import LogisticRegression

from pepnet import Predictor, SequenceInput, Output

from .common import parse_args, parse_allele_from_filename

from ..assembly import split_into_sequence_groups
from ..peptides import load_peptides_list_from_path
from ..dataset import Dataset

parser = ArgumentParser(
    description="Iterative training of binding-core predictor and overall predictor")

parser.add_argument("--hits", required=True)

parser.add_argument("--decoys-per-hit", type=int, default=10)

parser.add_argument("--binding-core-length", type=int, default=9)

parser.add_argument("--allele", help="Allele name to use in training datasets")

parser.add_argument(
    "--save-binding-core-training-csv",
    help="Optional path to CSV which saves training data for binding core predictor")


def build_training_set(sequence_groups, k, allele=None):
    ###
    # Train a binding core predictor from examples which have multiple
    # overlapping peptides.
    ###
    binding_core_training_peptides = []
    binding_core_training_labels = []
    binding_core_training_weights = []
    binding_core_training_group_ids = []
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
        alleles=[allele] * len(binding_core_training_peptides),
        peptides=binding_core_training_peptides,
        labels=binding_core_training_labels,
        weights=binding_core_training_weights,
        group_ids=binding_core_training_group_ids)


def encode(dataset, k):
    encoder = Encoder(variable_length_sequences=False)

    X = encoder.encode_onehot(
        dataset.peptides,
        max_peptide_length=k)
    # print("Generated one-hot matrix with shape: %s" % (X.shape,))
    n_samples, expect_k, expect_20 = X.shape
    assert expect_k == k, expect_k
    assert expect_20 == 20, expect_20
    X = X.reshape((n_samples, k * 20))
    y = np.array(dataset.labels)
    assert n_samples == len(y), len(y)
    weights = np.array(dataset.weights)
    return X, y, weights

def compute_positional_weight_matrix(seqs, k):
    alphabet = IUPAC.protein
    bio_seqs = [
        Seq(p[:k], alphabet=alphabet) for p in seqs
    ]
    m = motifs.create(
        bio_seqs,
        alphabet=alphabet)
    return m.counts.normalize(pseudocounts=0.5)

def apply_binding_core_classifier(clf, dataset, k):
    positive_dataset = dataset[dataset.labels]
    negative_dataset = dataset[~dataset.labels]
    group_ids_to_peptides = defaultdict(list)
    for (peptide, group_id) in zip(positive_dataset.peptides, positive_dataset.group_ids):
        group_ids_to_peptides[group_id].append(peptide)
    kept_positive_group_ids = []
    kept_positive_peptides = []
    for group_id, peptides in group_ids_to_peptides.items():
        X_group = Encoder(variable_length_sequences=False).encode_onehot(
            peptides, max_peptide_length=k)
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

def score_peptides(pssm, peptides):
    scores = []

    for peptide in peptides:
        for i, aa in enumerate(peptide):
            scores.append(pssm[aa][i])
    return np.mean(scores)

def iterative_classifier_training(
        dataset, k, max_iters=1000, nn=True):
    pwm = compute_positional_weight_matrix(
        seqs=dataset[dataset.labels].peptides,
        k=k)
    print("-- Consensus: %s" % (pwm.consensus,))
    X, y, weights = encode(dataset, k=k)

    clf = LogisticRegression(penalty="l1", C=0.1).fit(X, y)
    best_score = 0
    for i in range(max_iters):
        current_dataset = apply_binding_core_classifier(
            clf,
            dataset,
            k=k)
        pwm = compute_positional_weight_matrix(
            seqs=current_dataset[current_dataset.labels].peptides,
            k=k)
        pssm = pwm.log_odds()

        selected_binding_cores = current_dataset[current_dataset.labels].peptides
        score = score_peptides(pssm, selected_binding_cores)
        print("-- Consensus: %s (Score=%0.4f)" % (pwm.consensus, score))

        if score > best_score and i < max_iters - 1:
            best_score = score
            X, y, weights = encode(current_dataset, k=k)
            clf = LogisticRegression(penalty="l1", C=0.1).fit(X, y)

        else:
            print("-- Done!")
            break

    print("Final binding cores")
    for peptide in selected_binding_cores:
        print(peptide)

    return clf

def main(args_list=None):
    if not args_list:
        args_list = sys.argv[1:]
    args = parse_args(parser, args_list)
    allele = args.allele
    if not allele:
        allele = parse_allele_from_filename(args.hits)
    if not allele:
        raise ValueError("Could not determine allele name, specify using --allele")

    hits = load_peptides_list_from_path(args.hits)
    sequence_groups = split_into_sequence_groups(hits)

    k = args.binding_core_length
    dataset = build_training_set(sequence_groups, k=k, allele=allele)
    print(dataset)
    iterative_classifier_training(dataset=dataset, k=k)

