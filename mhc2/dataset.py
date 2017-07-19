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
from collections import OrderedDict, defaultdict
from serializable import Serializable

from six import string_types
import numpy as np
import pandas as pd

from .mhc_names import normalize_mhc_name
from .sequence_group import SequenceGroup
from .assembly import assemble_into_sequence_groups
from .peptides import load_allele_to_peptides_dict

class DatasetRow(Serializable):
    def __init__(self, allele, peptide, label, weight, contig, binding_core):
        self.allele = allele
        self.peptide = peptide
        self.label = label
        self.weight = weight
        self.contig = contig
        self.binding_core = binding_core

class Dataset(Serializable):
    """
    Training data for one or more alleles of Class II MHC peptide presentation
    """
    def _validate_and_normalize(
            self,
            alleles,
            peptides,
            labels,
            weights,
            contigs,
            binding_cores):
        if isinstance(peptides, string_types):
            peptides = [peptides]

        n_peptides = len(peptides)

        if alleles is None:
            alleles = [None] * n_peptides
        elif isinstance(alleles, string_types):
            alleles = [normalize_mhc_name(alleles)] * n_peptides
        else:
            alleles = [
                (normalize_mhc_name(allele) if allele  else None)
                for allele in alleles
            ]

        if n_peptides != len(alleles):
            raise ValueError(
                "Length mismatch between peptides (%d) and alleles (%d)" % (
                    n_peptides,
                    len(alleles)))

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

        if weights is None:
            # if weights aren't given then every element has the same
            # weight
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

        if contigs is None:
            contigs = peptides
        elif len(contigs) != len(peptides):
            raise ValueError("Expected %d contigs but got %d" % (
                len(peptides),
                len(contigs)))

        if binding_cores is None:
            binding_cores = peptides
        elif len(binding_cores) != len(peptides):
            raise ValueError("Expected %d binding cores but got %d" % (
                len(peptides),
                len(binding_cores)))
        rows = []
        for a, p, l, w, contig, core in zip(
                alleles,
                peptides,
                labels,
                weights,
                contigs,
                binding_cores):
            rows.append(DatasetRow(
                allele=a,
                peptide=p,
                label=l,
                weight=w,
                contig=contig,
                binding_core=core))
        return rows

    def __init__(
            self,
            alleles=None,
            peptides=[],
            labels=None,
            weights=None,
            contigs=None,
            binding_cores=None,
            extra_rows=[]):
        self.rows = self._validate_and_normalize(
                alleles=alleles,
                peptides=peptides,
                labels=labels,
                weights=weights,
                contigs=contigs,
                binding_cores=binding_cores)
        self.rows.extend(extra_rows)

    @classmethod
    def from_rows(cls, rows):
        return Dataset(extra_rows=rows)

    @property
    def peptides(self):
        return [row.peptide for row in self.rows]

    @property
    def alleles(self):
        return [row.allele for row in self.rows]

    @property
    def labels(self):
        return np.array([row.label for row in self.rows])

    @property
    def weights(self):
        return np.array([row.weight for row in self.rows])

    @property
    def contigs(self):
        return [row.contig for row in self.rows]

    @property
    def binding_cores(self):
        return [row.binding_core for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if len(self) != len(other):
            return False
        return all([
            ri == rj
            for (ri, rj)
            in zip(self.rows, other.rows)])

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "<Dataset with %d elements>\n%s" % (len(self), self.to_dataframe())

    def __repr__(self):
        return str(self)

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            new_rows = self.rows[indices]
        else:
            index_array = np.array(indices)
            if index_array.dtype == "bool":
                # replace boolean mask with indices of True entries
                index_array = np.where(index_array)[0]
            new_rows = [self.rows[i] for i in index_array]

        return Dataset.from_rows(new_rows)

    @classmethod
    def _check_type(cls, maybe_dataset):
        if maybe_dataset.__class__ is not cls:
            raise TypeError("Expected %s : %s to be %s" % (
                maybe_dataset, type(maybe_dataset), cls))

    @classmethod
    def make_empty(cls):
        return cls.from_rows([])

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

    def unique_contigs(self):
        return set(self.contigs)

    def unique_alleles(self):
        return set(self.alleles)

    def unique_peptides(self):
        return set(self.peptides)

    def combine(self, other):
        self._check_type(other)
        return self.from_rows(self.rows + other.rows)

    @classmethod
    def concat(cls, datasets):
        combined = cls.make_empty()
        for dataset in datasets:
            combined = combined.combine(dataset)
        return combined

    def to_dict(self):
        return OrderedDict([
            ("peptide", self.peptides),
            ("allele", self.alleles),
            ("label", self.labels),
            ("weight", self.weights),
            ("contig", self.contigs),
            ("binding_core", self.binding_cores),
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
        contigs = None
        binding_cores = None

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

        for name in ["binding_core", "binding_cores", "core", "cores"]:
            if name in col_names:
                binding_cores = df[name]
                break

        for name in [
                "contig", "contigs", "group_contig", "group_contigs",
                "group", "groups"]:
            if name in col_names:
                contigs = df[name]
                break

        return Dataset(
            alleles=alleles,
            peptides=peptides,
            labels=labels,
            weights=weights,
            binding_cores=binding_cores,
            contigs=contigs)

    def to_csv(self, filename):
        self.to_dataframe().to_csv(filename, index=False)

    @classmethod
    def from_csv(cls, filename):
        return cls.from_dataframe(pd.read_csv(filename))

    @classmethod
    def from_allele_to_peptides_dict(cls, allele_to_peptides_dict, label=True):
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
        return cls.from_allele_to_peptides_dict(
            allele_to_peptides_dict, label=label)

    @classmethod
    def from_allele_to_sequence_groups_dict(cls, allele_to_sequence_groups_dict, label=True):
        datasets = []
        for (allele, sequence_groups) in allele_to_sequence_groups_dict.items():
            datasets.append(
                Dataset.from_sequence_groups(
                    sequence_groups, allele=allele, label=label))
        return Dataset.concat(datasets)

    @classmethod
    def from_peptides_text_file(cls, filename, label=True):
        return cls.from_peptides_text_files([filename], label=label)

    @classmethod
    def from_peptides_text_files(cls, filenames, label=True):
        allele_to_peptides_dict = load_allele_to_peptides_dict(filenames)
        datasets = []
        for allele, peptides in allele_to_peptides_dict.items():
            datasets.append(Dataset(alleles=allele, labels=label, peptides=peptides))
        return cls.concat(datasets)

    def groupby_allele_dict(self):
        allele_to_rows = defaultdict(list)
        for row in self.rows:
            allele_to_rows[row.allele].append(row)
        return {
            allele: Dataset.from_rows(rows)
            for (allele, rows)
            in allele_to_rows.items()
        }

    def groupby_allele(self):
        for allele, dataset in self.groupby_allele_dict().items():
            yield allele, dataset

    def groupby_peptide(self):
        for peptide in sorted(self.unique_peptides(), key=lambda p: (len(p), p)):
            mask = [p_i == peptide for p_i in self.peptides]
            yield (peptide, self[mask])

    def group_peptides_by_allele_and_contig_dict(self):
        result = defaultdict(list)
        for (peptide, allele, contig) in zip(self.peptides, self.alleles, self.contigs):
            result[(allele, contig)].append(peptide)
        return result

    def group_peptides_by_allele_and_contig(self):
        for (allele, contig), peptides in self.group_peptides_by_allele_and_contig_dict():
            yield (allele, contig), peptides

    def group_rows_by_contig_dict(self):
        result = defaultdict(list)
        for row in self.rows:
            result[row.contig].append(row)
        return result

    def group_rows_by_contig(self):
        for contig, rows in self.group_rows_by_contig_dict().items():
            yield contig, rows

    def group_rows_by_allele_and_label_dict(self):
        row_groups =  defaultdict(list)
        for row in self.rows:
            key = (row.allele, row.label)
            row_groups[key].append(row)
        return row_groups

    def group_rows_by_allele_and_label(self):
        for (key, rows) in self.group_rows_by_allele_and_label_dict().items():
            yield key, rows

    def group_by_allele_and_label_dict(self):
        return {
            key: Dataset.from_rows(rows)
            for (key, rows) in self.group_rows_by_allele_and_label()
        }

    def group_by_allele_and_label(self):
        for key, dataset in self.group_by_allele_and_label_dict().items():
            yield key, dataset

    def group_peptides_by_allele_and_label_dict(self):
        return {
            key: [row.peptide for row in rows]
            for (key, rows) in self.group_rows_by_allele_and_label()
        }

    def group_peptides_by_allele_and_label(self):
        for key, peptides in self.group_peptides_by_allele_and_label_dict().items():
            yield key, peptides

    def allele_and_contig_pairs_to_unique_ids(self):
        d = self.group_peptides_by_allele_and_contig_dict()
        return {key: i for i, key in enumerate(d.keys())}

    @property
    def group_ids(self):
        """
        Returns list of group_ids of same length as peptides
        """
        if hasattr(self, '_group_ids'):
            return self._group_ids
        d = self.allele_and_contig_pairs_to_unique_ids()
        return [
            d[(allele, contig)]
            for (allele, contig)
            in zip(self.alleles, self.contigs)
        ]

    def assemble_contigs(self):
        """
        Ignoring group contigs and use overlap assembly to construct
        longer sequences containing peptides for each allele.
        """
        datasets = []
        for (allele, label), peptides in self.group_peptides_by_allele_and_label():
            sequence_groups = assemble_into_sequence_groups(peptides)
            datasets.append(Dataset.from_sequence_groups(
                sequence_groups, allele=allele, label=label))
        return Dataset.concat(datasets)

    @classmethod
    def from_sequence_groups(
            cls,
            sequence_groups,
            allele=None,
            label=True):
        """
        Create flattened Dataset from list of SequenceGroup objects
        """
        rows = []

        def sort_by_increasing_len(s):
            return len(s)

        for group in sequence_groups:
            binding_cores = group.binding_cores
            if len(binding_cores) > 1:
                binding_cores = sorted(binding_cores, key=sort_by_increasing_len)
            for c in sorted(group.children, key=lambda x: (len(x), x)):
                binding_core = None
                for seq in group.binding_cores:
                    if seq in c:
                        binding_core = seq
                        break
                row = DatasetRow(
                    allele=allele,
                    peptide=c,
                    contig=group.contig,
                    binding_core=binding_core,
                    label=True,
                    weight=1.0 / len(group.children))
                rows.append(row)
        return cls.from_rows(rows)

    def to_sequence_groups(self):
        """
        Returns list of SequenceGroup objects, throws away
        weight, allele, and label for each row.
        """
        sequence_groups = []
        for (contig, rows) in self.group_rows_by_contig():
            children = [r.peptide for r in rows]
            binding_cores = {r.binding_core for r in rows if r.binding_core}
            # leaf sequences aren't contained in other peptides
            leaves = {
                seq for seq in children
                if sum(seq in other_seq for other_seq in children) == 1
            }
            sequence_groups.append(
                SequenceGroup(
                    contig=contig,
                    children=children,
                    leaves=leaves,
                    binding_cores=binding_cores))
        return sequence_groups
