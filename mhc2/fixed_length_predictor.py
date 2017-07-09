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

from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import numpy as np

from Bio import motifs
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq

from pepnet import Encoder

from .dataset import Dataset



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

def compute_positional_weight_matrix(seqs, length=None):
    if not length:
        length = min(len(s) for s in seqs)
    alphabet = IUPAC.protein
    bio_seqs = [
        Seq(p[:length], alphabet=alphabet) for p in seqs
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

def score_peptides(peptides):
    pwm = compute_positional_weight_matrix(peptides)
    pssm = pwm.log_odds()
    scores = []
    for peptide in peptides:
        for i, aa in enumerate(peptide):
            scores.append(pssm[aa][i])
    return np.mean(scores)

def train_binding_core_classifier(
        dataset,
        binding_core_size=9,
        max_iters=1000,
        make_classifier=lambda: LogisticRegression(penalty="l1", C=0.1)):
    """
    Iterative training of a fixed length binding core classifier which is then
    used to refine the training data.
    """
    # create initial classifer by treating all 9mer subsequences as equally
    # likely binding cores
    X, y, weights = encode(dataset, k=binding_core_size)
    clf = make_classifier().fit(X, y)
    best_score = 0
    for i in range(max_iters):
        current_dataset = apply_binding_core_classifier(
            clf,
            dataset,
            k=binding_core_size)
        selected_binding_cores = current_dataset[current_dataset.labels].peptides
        score = score_peptides(selected_binding_cores)
        if score > best_score:
            best_score = score
            X, y, weights = encode(current_dataset, k=binding_core_size)
            clf = make_classifier().fit(X, y)
        else:
            break
    return clf, selected_binding_cores
