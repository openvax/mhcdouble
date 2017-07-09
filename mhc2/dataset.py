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

import math
from collections import OrderedDict

from six import string_types
import numpy as np
import pandas as pd

from .mhc_names import normalize_mhc_name, normalize_mhc_names

class Dataset(object):
    """
    Training data for one or more alleles of Class II MHC peptide presentation
    """
    def _validate_and_normalize(
            self, alleles, peptides, labels, weights, group_ids):
        if isinstance(peptides, string_types):
            peptides = [peptides]

        n_peptides = len(peptides)

        if n_peptides != len(alleles):
            raise ValueError(
                "Length mismatch between peptides (%d) and alleles (%d)" % (
                    n_peptides,
                    len(alleles)))

        if isinstance(alleles, string_types):
            alleles = [normalize_mhc_name(alleles)] * n_peptides
        else:
            alleles = normalize_mhc_names(alleles)

        if labels is None:
            # if labels are not given then assume that the dataset is
            # comprised of "hits"
            labels = np.ones(n_peptides, dtype="bool")
        elif isinstance(labels, (float, int, bool)):
            labels = np.array([labels] * n_peptides, dtype="bool")
        elif n_peptides != len(labels):
            raise ValueError(
                "Length mismatch between peptides (%d) and labels (%d)" % (
                    n_peptides,
                    len(labels)))
        else:
            labels = np.array(labels, dtype="bool")

        if group_ids is None:
            # if sequence groups aren't given then put every
            # peptide in a singleton group
            group_ids = np.arange(n_peptides, dtype="int32")
        elif isinstance(group_ids, (float, int, bool)):
            group_ids = np.array([group_ids] * n_peptides, dtype="int32")
        elif n_peptides != len(group_ids):
            raise ValueError(
                "Length mismatch between peptides (%d) and groups (%d)" % (
                    n_peptides,
                    len(group_ids)))
        else:
            group_ids = np.array(group_ids, dtype="int32")

        if weights is None:
            # if weights aren't given then every element has the same
            # weight
            # TODO: use group_ids and labels to weigh samples inversely with
            # group size (separately for positive vs. negative labels)
            weights = np.ones(n_peptides, dtype="float32")
        elif isinstance(weights, (float, int)):
            weights = np.array([weights] * n_peptides, dtype="float32")
        elif n_peptides != len(weights):
            raise ValueError(
                "Length mismatch between peptides (%d) and weights (%d)" % (
                    n_peptides,
                    len(weights)))
        else:
            weights = np.array(weights, dtype="float32")

        return alleles, peptides, labels, weights, group_ids

    def __init__(self, alleles, peptides, labels=None, weights=None, group_ids=None):
        alleles, peptides, labels, weights, group_ids = \
            self._validate_and_normalize(
                alleles=alleles,
                peptides=peptides,
                labels=labels,
                weights=weights,
                group_ids=group_ids)
        self.alleles = alleles
        self.peptides = peptides
        self.labels = labels
        self.weights = weights
        self.group_ids = group_ids

    def __len__(self):
        return len(self.peptides)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if len(self) != len(other):
            return False
        other_dict = other.to_dict()
        for (column_name, values) in self.to_dict().items():
            other_values = other_dict[column_name]
            for (x, y) in zip(values, other_values):
                if x != y:
                    print(x, y)
                    return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "<Dataset with %d elements>\n%s" % (len(self), self.to_dataframe())

    def __repr__(self):
        return str(self)

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            peptides = self.peptides[:indices]
            alleles = self.alleles[:indices]
            # for other fields we want to pull out the indices
            # implied by a slice
            index_array = np.arange(*indices.indices(len(self)))
        else:
            index_array = np.array(indices)
            if index_array.dtype == "bool":
                # replace boolean mask with indices of True entries
                index_array = np.where(index_array)[0]
            peptides = [self.peptides[i] for i in index_array]
            alleles = [self.alleles[i] for i in index_array]

        return Dataset(
            alleles=alleles,
            peptides=peptides,
            labels=self.labels[index_array],
            weights=self.weights[index_array],
            group_ids=self.group_ids[index_array])

    @classmethod
    def _check_type(cls, maybe_dataset):
        if maybe_dataset.__class__ is not cls:
            raise TypeError("Expected %s : %s to be %s" % (
                maybe_dataset, type(maybe_dataset), cls))

    @classmethod
    def make_empty(cls):
        return cls(alleles=[], peptides=[], labels=[], weights=[], group_ids=[])

    def split(self, train_idx, test_idx):
        dataset_train = self[train_idx]
        dataset_test = self[test_idx]
        return dataset_train, dataset_test

    def random_subset(self, fraction=0.5):
        subset, _ = self.random_split(fraction=fraction)
        return subset

    def random_split(self, fraction=0.5):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        n_left = int(math.ceil(len(self) * fraction))
        subset_left = self[indices[:n_left]]
        subset_right = self[indices[n_left:]]
        return subset_left, subset_right

    def shuffle(self):
        left, right = self.random_split(fraction=1.0)
        if len(left) != len(self):
            raise ValueError("Expected %d elements but got %d after shuffle" % (
                len(self),
                len(left)))
        assert len(right) == 0
        return left

    def distinct_sequence_groups(self):
        return set(self.group_ids)

    def distinct_alleles(self):
        return set(self.alleles)

    def distinct_peptides(self):
        return set(self.peptides)

    def groupby_allele(self):
        for allele in self.distinct_alleles():
            mask = [a_i == allele for a_i in self.alleles]
            yield (allele, self[mask])

    def groupby_peptide(self):
        for peptide in self.distinct_peptides():
            mask = [p_i == peptide for p_i in self.peptides]
            yield (peptide, self[mask])

    def max_group_id(self):
        if len(self) > 0:
            return
        else:
            return -1

    def combine(self, other, preserve_group_ids=False):
        self._check_type(other)
        peptides = list(self.peptides) + list(other.peptides)
        alleles = list(self.alleles) + list(other.alleles)
        labels = np.concatenate([self.labels, other.labels])
        weights = np.concatenate([self.weights, other.weights])
        if not preserve_group_ids and len(self) > 0:
            offset = self.group_ids.max() + 1
            other_group_ids = offset + other.group_ids
        else:
            other_group_ids = other.group_ids
        group_ids = np.concatenate([self.group_ids, other_group_ids])
        return Dataset(
            alleles=alleles,
            peptides=peptides,
            labels=labels,
            weights=weights,
            group_ids=group_ids)

    @classmethod
    def concat(cls, datasets, preserve_group_ids=False):
        combined = cls.make_empty()
        for dataset in datasets:
            combined = combined.combine(dataset, preserve_group_ids=preserve_group_ids)
        return combined

    def to_dict(self):
        return OrderedDict([
            ("peptide", self.peptides),
            ("allele", self.alleles),
            ("label", self.labels),
            ("weight", self.weights),
            ("group_id", self.group_ids)
        ])

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict())

    @classmethod
    def from_dataframe(cls, df):
        col_names = set(df.columns)
        peptides = None
        alleles = None
        labels = None
        weights = None
        group_ids = None

        for name in ["peptide", "peptides", "sequence", "sequences", "seq", "seqs"]:
            if name in col_names:
                peptides = list(df[name])
                break
        if peptides is None:
            raise ValueError("Missing peptides column, available: %s" % (col_names,))

        for name in ["mhc", "allele", "alleles"]:
            if name in col_names:
                alleles = list(df[name])
                break
        if alleles is None:
            raise ValueError("Missing alleles column, available: %s" % (col_names,))

        for name in ["label", "labels", "hit", "hits"]:
            if name in col_names:
                labels = np.array(df[name])
                break

        if labels is None:
            labels = np.array([True] * len(peptides))

        for name in ["weight", "weights", "sample_weight", "sample_weights"]:
            if name in col_names:
                weights = np.array(df[name])
                break

        for name in ["group_id", "group_ids", "group", "groups"]:
            if name in col_names:
                group_ids = np.array(df[name])
                break

        return Dataset(
            alleles=alleles,
            peptides=peptides,
            labels=labels,
            weights=weights,
            group_ids=group_ids)

    def to_csv(self, filename):
        self.to_dataframe().to_csv(filename, index=False)

    @classmethod
    def from_csv(cls, filename):
        return cls.from_dataframe(pd.read_csv(filename))

    @classmethod
    def from_allele_dict(cls, allele_to_peptides_dict, label=True):
        """
        Create Dataset from allele->peptide dictionary of hits
        """
        alleles = []
        peptides = []
        for allele, allele_peptides in allele_to_peptides_dict.items():
            alleles.extend([allele] * len(allele_peptides))
            peptides.extend(allele_peptides)
        labels = [label] * len(peptides)
        return cls(alleles=alleles, peptides=peptides, labels=labels)

    @classmethod
    def from_excel(cls, filename, label=True):
        df = pd.read_excel(filename)
        allele_to_peptides_dict = {}
        for allele in df.columns:
            allele_to_peptides_dict[allele] = [
                s.upper() for s in df[allele]
                if isinstance(s, str) and len(s) > 0 and ("X" not in s)
            ]
        return cls.from_allele_dict(
            allele_to_peptides_dict, label=label)
