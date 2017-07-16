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
from collections import Counter
from pyensembl import ensembl_grch38

from .dataset import Dataset
from .sequence_group import SequenceGroup

class PeptideGenerator(object):
    _genome_to_sequence = {}

    @classmethod
    def full_proteome_sequence(cls, genome=ensembl_grch38):
        if genome not in cls._genome_to_sequence:
            # make a single string out of all the proteins
            concatenated_proteins = "".join(
                ensembl_grch38.protein_sequences.fasta_dictionary.values())
            cls._genome_to_sequence[genome] = concatenated_proteins
        else:
            concatenated_proteins = cls._genome_to_sequence[genome]
        return concatenated_proteins

    def generate_independent_peptides(
            self,
            n_peptides,
            desired_lengths,
            desired_length_probs=None,
            exclude_set=set([]),
            filter_bad_chars=True,
            fudge_factor=1.2):
        """
        Parameters
        ----------
        n_peptides : int
            Number of peptides

        desired_lengths : int, list of int, or dict from int to float
            If just one integer then all returned peptides will be of this
            length. If a list, then each length will get the same
            probability.

        exclude_set : set
            Set of peptides to exclude from results

        filter_bad_chars : bool
            Exclude peptides with "X", "*", or "U" (selenocysteine)

        fudge_factor : float
            Degree to which we oversample to account for filtered/excluded peptides

        Return random subsequences from the proteome with one caveat:
            - some peptides may span the boundary between two proteins
        """
        if isinstance(desired_lengths, int):
            desired_lengths = [desired_lengths]
        else:
            desired_lengths = list(desired_lengths)

        n_unique_lengths = len(desired_lengths)

        if desired_length_probs is None:
            equal_weight = 1.0 / n_unique_lengths
            desired_length_probs = [equal_weight] * n_unique_lengths

        max_samples = int(fudge_factor * n_peptides)

        random_lengths = np.random.choice(
            desired_lengths,
            size=max_samples,
            replace=True,
            p=desired_length_probs)
        full_proteome_sequence = self.full_proteome_sequence()
        max_pos = len(full_proteome_sequence) - 1

        random_positions = np.random.randint(0, max_pos, max_samples)

        results = []
        for l, pos in zip(random_lengths, random_positions):
            if len(results) >= n_peptides:
                break
            s = full_proteome_sequence[pos:pos + l]
            if (s in exclude_set) or (len(s) != l) or ("*" in s) or ("X" in s) or ("U" in s):
                continue
            results.append(s)

        if len(results) < n_peptides:
            raise ValueError("Unable to make sufficient number of decoys (%d < %d)" % (
                len(results), n_peptides))

        return results

peptide_generator = PeptideGenerator()

def generate_independent_decoy_list_from_proteome(
        n_decoys,
        positive_peptides=[],
        match_length_distribution=False,
        min_length=None,
        max_length=None):
    original_set = set(positive_peptides)
    unique_lengths = sorted({len(p) for p in positive_peptides})
    if match_length_distribution:
        length_counts = Counter()
        for p in positive_peptides:
            length_counts[len(p)] += 1
        unique_length_counts = [
            length_counts[l] for l in unique_lengths
        ]
        decoy_lengths = unique_lengths
        decoy_lengths_probs = (
            unique_length_counts.astype("float32") / unique_length_counts.sum())

    else:
        if min_length is None:
            min_length = min(unique_lengths)
        if max_length is None:
            max_length = max(unique_lengths)
        decoy_lengths = np.arange(min_length, max_length + 1)
        decoy_lengths_probs = None

    return peptide_generator.generate_independent_peptides(
            n_peptides=n_decoys,
            desired_lengths=decoy_lengths,
            desired_length_probs=decoy_lengths_probs,
            exclude_set=original_set)

def augment_dataset_with_decoys(
        hit_dataset,
        decoys_per_hit,
        min_decoy_length=None,
        max_decoy_length=None):
    n_hits = len(hit_dataset)
    n_decoys = int(n_hits * decoys_per_hit)
    decoy_peptides = generate_independent_decoy_list_from_proteome(
        n_decoys=n_decoys,
        positive_peptides=hit_dataset.peptides,
        min_length=min_decoy_length,
        max_length=max_decoy_length)

    assert n_decoys == len(decoy_peptides)
    decoy_mhc_alleles = list(np.random.choice(hit_dataset.alleles, size=n_decoys))

    sum_hit_weights = hit_dataset.weights.sum()
    hits_to_decoys = sum_hit_weights / float(n_decoys)
    decoy_weight = min(1.0, hits_to_decoys)
    decoy_dataset = Dataset(
        peptides=decoy_peptides,
        alleles=decoy_mhc_alleles,
        labels=0,
        weights=decoy_weight)
    return hit_dataset.combine(decoy_dataset, preserve_group_ids=False)

def generate_decoy_sequence_groups(
            n_decoy_loci,
            min_decoys_per_locus=1,
            max_decoys_per_locus=10,
            binding_core_length=9,
            contig_length=40):
    sequence_groups = []
    contigs = generate_independent_decoy_list_from_proteome(
        n_decoys=n_decoy_loci,
        min_length=contig_length,
        max_length=contig_length)
    half_idx = contig_length // 2

    max_decoys = int(np.ceil(n_decoy_loci * max_decoys_per_locus))
    start_indices = np.random.randint(
        low=0,
        high=half_idx,
        size=max_decoys)
    end_indices = np.random.randint(
        low=half_idx + binding_core_length,
        high=contig_length,
        size=max_decoys)
    offset = 0
    count_per_locus = np.random.randint(
        low=min_decoys_per_locus,
        high=max_decoys_per_locus + 1,
        size=n_decoy_loci)
    for i, contig in enumerate(contigs):
        binding_core = contig[half_idx:half_idx + binding_core_length]
        n_peptides = count_per_locus[i]
        children = []
        for _ in range(n_peptides):
            start = start_indices[offset]
            end = end_indices[offset]
            children.append(contig[start:end])
            offset += 1

        sequence_groups.append(SequenceGroup(
            contig=contig,
            binding_cores=[binding_core],
            leaves=[binding_core],
            children=set(children)))
    return sequence_groups

