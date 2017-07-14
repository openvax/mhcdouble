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

from .common import all_splits
from .sequence_group import SequenceGroup

MIN_ASSEMBLY_OVERLAP_SIZE = 5
MIN_BINDING_CORE_SIZE = 7

def naive_left_overlap(s, t, min_overlap_size=MIN_ASSEMBLY_OVERLAP_SIZE):
    """
    Returns number of characters on right side of input `s`
    that are equal to the left side of input sequence `t`.
    """
    if s[-min_overlap_size:] not in t:
        # if the smallest substring of `s` isn't in `t` then
        # stop before we try to find the overlapping piece
        return 0

    if t.startswith(s):
        return len(s)

    # search over all possible rightmost pieces of `s` which might
    # be the same as the leftmost piece of `t`
    for i in range(1, len(s) - min_overlap_size):
        candidate = s[i:]
        if candidate == t[:len(candidate)]:
            return len(candidate)
    return 0

def find_overlaps(sequences, min_overlap_size=MIN_ASSEMBLY_OVERLAP_SIZE):
    left_overlap_dict = defaultdict(list)
    for s in sorted(sequences):
        for t in sequences:
            if s == t:
                continue
            n_overlap = naive_left_overlap(s, t)
            if n_overlap >= min_overlap_size:
                left_overlap_dict[s].append(t)
    return left_overlap_dict

def find_contained_sequences(contigs, short_sequences=None):
    """
    Returns two objects:
        1) dictionary which maps each sequence to list of "child" sequences
           contained within it
        2) dictionary which maps each sequence to list of "parent" sequences
           which contain it
    """
    if short_sequences is None:
        short_sequences = contigs
    child_dict = defaultdict(list)
    parent_dict = defaultdict(list)
    for x in short_sequences:
        for y in contigs:
            if x != y:
                if x in y:
                    child_dict[y].append(x)
                    parent_dict[x].append(y)
    return (child_dict, parent_dict)


def find_root_sequences(
        sequences, child_dict, parent_dict):
    """
    Parameters
    ----------
    child_dict : dict
        Maps each sequence to list of its contained children

    parent_dict : dict
        Maps each sequence to list of parent sequences which contain it

    Returns "root" sequences which have children but no parents.
    """
    sequences_with_children = set(child_dict.keys())
    sequences_with_parents = set(parent_dict.keys())
    root_sequences = sequences_with_children.difference(sequences_with_parents)
    singleton_sequences = set(sequences).difference(
        sequences_with_children.union(
            sequences_with_parents))
    return root_sequences.union(singleton_sequences)

def assemble_overlapping_sequences(sequences):
    """
    Find overlap between all input sequences and assemble new
    longer contigs where possible.
    """
    left_overlap_dict = find_overlaps(sequences)
    new_candidate_set = set([])
    for (left_seq, right_seqs) in left_overlap_dict.items():
        right_seq = max(right_seqs, key=lambda x: len(x))
        n_overlap = naive_left_overlap(left_seq, right_seq)
        assert n_overlap is not None and n_overlap > 0
        combined = left_seq[:-n_overlap] + right_seq
        new_candidate_set.add(combined)
    return new_candidate_set

def add_sequences_not_contained(new_sequences, previous_sequences):
    """
    If any of the previous iteration's sequences aren't contained in
    assembled "new" sequences, then add them to the result set.
    """
    combined_sequences = set(new_sequences)
    for s in previous_sequences:
        found_containing_seq = False
        for t in new_sequences:
            if s in t:
                found_containing_seq = True
                break
        if not found_containing_seq:
            combined_sequences.add(s)
    return combined_sequences

def assemble_sequences(sequences, min_overlap_size=MIN_ASSEMBLY_OVERLAP_SIZE):
    """
    Given a collection of sequences, construct a smaller set of sequences which
    contain all of the inputs by overlap assembly.
    """
    candidate_set = set(sequences)
    iteration = 1
    while iteration < 100:
        print("[Assembly] Iteration %d: # candidates = %d" % (
            iteration,
            len(candidate_set)))
        child_dict, parent_dict = find_contained_sequences(candidate_set)
        root_sequences = \
            find_root_sequences(
                sequences=candidate_set,
                child_dict=child_dict,
                parent_dict=parent_dict)
        new_candidate_set = assemble_overlapping_sequences(root_sequences)
        new_candidate_set = add_sequences_not_contained(
            new_sequences=new_candidate_set,
            previous_sequences=root_sequences)
        if candidate_set == new_candidate_set:
            break
        else:
            iteration += 1
            candidate_set = new_candidate_set
    if iteration >= 100:
        print("Assembly failed to converge!")
    return candidate_set

def intersect_sequences(sequences, min_overlap_size=MIN_BINDING_CORE_SIZE):
    """
    Attempt to find minimal epitopes by intersecting all the leaves
    but if this fails then just fall back on using the leaf sequences
    without modification
    """

    result = list(sequences)[0]
    if len(sequences) == 1:
        return result

    for i, s in enumerate(sequences):
        n_left_overlap = naive_left_overlap(
            s,
            result,
            min_overlap_size=min_overlap_size)
        n_right_overlap = naive_left_overlap(
            result,
            s,
            min_overlap_size=min_overlap_size)
        if n_left_overlap > 0:
            result = result[:n_left_overlap]
        elif n_right_overlap > 0:
            result = result[-n_right_overlap:]
        else:
            return None
    return result

def binding_cores_from_leaf_sequences(
        sequences, min_overlap_size=MIN_BINDING_CORE_SIZE):

    intersection = intersect_sequences(sequences, min_overlap_size=min_overlap_size)
    if intersection is not None:
        return [intersection]
    best_candidate = sequences
    # try spitting sequences into two groups, assuming that there are two
    # distinct binding cores
    for s1, s2 in all_splits(sequences, nonempty=True):
        i1 = intersect_sequences(s1, min_overlap_size=min_overlap_size)
        i2 = intersect_sequences(s2, min_overlap_size=min_overlap_size)
        if i1 is not None and i2 is not None:
            return [i1, i2]
        if i1 is not None:
            candidate = [i1] + list(s2)
        elif i2 is not None:
            candidate = [i2] + list(s1)
        if len(candidate) < len(best_candidate):
            best_candidate = candidate
    return best_candidate

def assemble_and_assign_to_sequence_groups(
        sequences,
        min_overlap_size=MIN_ASSEMBLY_OVERLAP_SIZE,
        min_binding_core_size=MIN_BINDING_CORE_SIZE):
    original_sequence_set = set(sequences)
    assembled_sequences = assemble_sequences(
        original_sequence_set,
        min_overlap_size=min_overlap_size)

    (child_dict, parent_dict) = find_contained_sequences(
        contigs=assembled_sequences.union(original_sequence_set))
    groups = []
    for s in assembled_sequences:

        children = list(child_dict.get(s, []))
        if s in original_sequence_set:
            children.append(s)
        leaves = set([])
        for c in children:
            if c not in child_dict:
                leaves.add(c)
        binding_cores = binding_cores_from_leaf_sequences(leaves)
        group = SequenceGroup(
            contig=s,
            children=children,
            leaves=leaves,
            binding_cores=binding_cores)
        groups.append(group)
    return groups
