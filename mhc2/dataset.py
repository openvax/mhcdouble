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
from collections import OrderedDict, defaultdict, Counter
from serializable import Serializable

from six import string_types, integer_types
import numpy as np
import pandas as pd
from mhcnames import normalize_allele_name

from .sequence_group import SequenceGroup
from .assembly import assemble_into_sequence_groups
from .peptides import load_allele_to_peptides_dict


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
            binding_cores,
            extra_columns):
        """
        Given list or array values for the four core fields
        (allele, peptide, label, weight), as well as a dictionary
        of extra columns, checks to make sure that values are of
        correct types and lengths and then returns a DataFrame
        of normalized values.
        """
        if isinstance(peptides, string_types):
            peptides = [peptides]

        n_peptides = len(peptides)

        if alleles is None:
            alleles = [None] * n_peptides
        elif isinstance(alleles, string_types):
            alleles = [normalize_allele_name(alleles)] * n_peptides
        else:
            alleles = [
                (normalize_allele_name(allele) if allele  else None)
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
            labels = np.array([labels] * n_peptides)
        elif n_peptides != len(labels):
            raise ValueError(
                "Length mismatch between peptides (%d) and labels (%d)" % (
                    n_peptides,
                    len(labels)))
        else:
            labels = np.array(labels)

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
        elif len(contigs) != n_peptides:
            raise ValueError("Expected %d contigs but got %d" % (len(contigs), n_peptides))

        if binding_cores is None:
            binding_cores = contigs
        elif len(binding_cores) != n_peptides:
            raise ValueError("Expected %d binding cores but got %d" % (
                len(binding_cores), n_peptides))

        result_dict = OrderedDict([])
        result_dict["allele"] = alleles
        result_dict["peptide"] = peptides
        result_dict["label"] = labels
        result_dict["weight"] = weights
        result_dict["contig"] = contigs
        result_dict["binding_core"] = binding_cores

        if extra_columns is None:
            extra_columns = {}
        else:
            assert isinstance(extra_columns, dict)
            extra_columns = extra_columns.copy()

        for (name, values) in extra_columns.items():
            scalar_types = integer_types + string_types + (bool, float)
            if isinstance(values, scalar_types):
                values = [values] * n_peptides
            elif values is None:
                values = [None] * n_peptides
            elif len(values) != n_peptides:
                raise ValueError(
                    "Expected %d values for feature '%s' but got %d" % (
                        n_peptides,
                        name,
                        len(values)))
            result_dict[name] = values
        return pd.DataFrame(result_dict)

    def __init__(
            self,
            alleles=None,
            peptides=[],
            labels=None,
            weights=None,
            contigs=None,
            binding_cores=None,
            extra_columns={}):
        df = self._validate_and_normalize(
            alleles=alleles,
            peptides=peptides,
            labels=labels,
            weights=weights,
            contigs=contigs,
            binding_cores=binding_cores,
            extra_columns=extra_columns)
        self.dataframe = df


    @classmethod
    def from_rows(cls, rows):
        return Dataset.from_dataframe(
            pd.DataFrame.from_records(rows))

    @property
    def rows(self):
        return [row for _, row in self.dataframe.iterrows()]

    @property
    def row_dicts(self):
        return [row.to_dict() for row in self.rows]

    def map_rows(self, fn, filter_fn=None):
        if filter_fn is None:
            new_rows = [fn(row) for row in self.rows]
        else:
            new_rows = [fn(row) for row in self.rows if filter_fn(row)]
        return self.from_rows(new_rows)

    @property
    def peptides(self):
        return list(self.dataframe.peptide)

    @property
    def alleles(self):
        return list(self.dataframe.allele)

    @property
    def labels(self):
        return np.array(self.dataframe.label)

    @property
    def weights(self):
        return np.array(self.dataframe.weight)

    @property
    def contigs(self):
        return list(self.dataframe.contig)

    @property
    def binding_cores(self):
        return list(self.dataframe.binding_core)

    def __len__(self):
        return len(self.dataframe)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        if len(self) != len(other):
            return False

        if set(self.dataframe.columns) != set(other.dataframe.columns):
            return False

        for name in self.dataframe.columns:
            x = self.dataframe[name]
            y = other.dataframe[name]
            x_null = x.isnull()
            y_null = y.isnull()
            if (x_null != y_null).any():
                return False
            if (x[~x_null] != y[~y_null]).any():
                return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "<Dataset with %d elements>\n%s" % (len(self), self.dataframe)

    def __repr__(self):
        return str(self)

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            sub_df = self.dataframe.iloc[indices]
        else:
            index_array = np.array(indices)
            if index_array.dtype == "bool":
                # replace boolean mask with indices of True entries
                index_array = np.where(index_array)[0]
            sub_df = self.dataframe.iloc[indices]
        return Dataset.from_dataframe(sub_df)

    @classmethod
    def _check_type(cls, maybe_dataset):
        if maybe_dataset.__class__ is not cls:
            raise TypeError("Expected %s : %s to be %s" % (
                maybe_dataset, type(maybe_dataset), cls))

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
        dataframes = [dataset.dataframe for dataset in datasets]
        return Dataset.from_dataframe(pd.concat(dataframes, ignore_index=True))

    def to_dict(self):
        d = OrderedDict([])
        for name in self.dataframe.columns:
            d[name] = list(self.dataframe[name])
        return d

    def to_dataframe(self):
        """
        Deprecated!
        """
        return self.dataframe

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

        # TODO: this makes sense is we're just loading a list of hits but
        # is stranger when dealing with affinity data
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
            contigs=contigs,
            binding_cores=binding_cores)

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

    def group_rows_by_allele_and_label_dict(self, label_threshold=None):
        row_groups =  defaultdict(list)
        for row in self.rows:
            if label_threshold:
                label = row.label >= label_threshold
            else:
                label = row.label
            key = (row.allele, label)
            row_groups[key].append(row)
        return row_groups

    def group_rows_by_allele_and_label(self, label_threshold=None):
        for (key, rows) in self.group_rows_by_allele_and_label_dict(
                label_threshold=label_threshold).items():
            yield key, rows

    def group_by_allele_and_label_dict(self, label_threshold=None):
        return {
            key: Dataset.from_rows(rows)
            for (key, rows) in self.group_rows_by_allele_and_label(
                    label_threshold=label_threshold)
        }

    def group_by_allele_and_label(self, label_threshold=None):
        for key, dataset in self.group_by_allele_and_label_dict(
                label_threshold=label_threshold).items():
            yield key, dataset

    def group_peptides_by_allele_and_label_dict(self, label_threshold=None):
        return {
            key: [row.peptide for row in rows]
            for (key, rows) in self.group_rows_by_allele_and_label(
                    label_threshold=label_threshold)
        }

    def group_peptides_by_allele_and_label(self, label_threshold=None):
        for key, peptides in self.group_peptides_by_allele_and_label_dict(
                label_threshold=label_threshold).items():
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

    def assemble_contigs(
            self,
            search_for_binding_cores=True,
            min_overlap_size=5):
        """
        Ignoring group contigs and use overlap assembly to construct
        longer sequences containing peptides for each allele.
        """
        datasets = []
        for (allele, label), peptides in self.group_peptides_by_allele_and_label():
            print(" -- Group allele=%s, label=%s" % (allele, label))
            sequence_groups = assemble_into_sequence_groups(
                peptides,
                search_for_binding_cores=search_for_binding_cores,
                min_overlap_size=min_overlap_size)
            datasets.append(Dataset.from_sequence_groups(
                sequence_groups,
                allele=allele,
                label=label))
        result = Dataset.concat(datasets)
        return result

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

        peptide_counts = Counter()
        for group in sequence_groups:
            for peptide in group.children:
                peptide_counts[peptide] += 1

        for group in sequence_groups:
            binding_cores = group.binding_cores
            if len(binding_cores) > 1:
                binding_cores = sorted(binding_cores, key=sort_by_increasing_len)
            sorted_peptides = sorted(group.children, key=lambda x: (len(x), x))
            group_size = len(group.children)
            for peptide in sorted_peptides:
                binding_core = None
                for seq in group.binding_cores:
                    if seq in peptide:
                        binding_core = seq
                        break
                # TODO: figure out a weight for each peptide
                # that makes each group have weights that sum to 1
                # but also gives lesser weight to peptides
                # that appear across multiple groups.
                #
                # Unclear if it's possible to make arbitary
                # weight matrices doubly-stochastic
                #
                # For now erring on the side of having the groups sum to less
                # than 1 and groups get more or less weight depending on
                # how distinctive their peptides are.

                peptide_count = peptide_counts[peptide]
                row = OrderedDict(
                    allele=allele,
                    peptide=peptide,
                    contig=group.contig,
                    binding_core=binding_core,
                    label=True,
                    weight=1.0 / (peptide_count * group_size),
                    n_groups_containing_peptide=peptide_count,
                    n_peptides_in_group=group_size,)
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
