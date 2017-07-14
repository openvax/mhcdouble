from mhc2.decoys import (
    generate_independent_decoy_list_from_proteome,
    generate_decoy_sequence_groups
)
from nose.tools import eq_

def test_generate_independent_decoy_list_from_proteome():
    decoys = generate_independent_decoy_list_from_proteome(
        n_decoys=2000,
        min_length=10,
        max_length=15)
    eq_(len(decoys), 2000)
    eq_(min(len(p) for p in decoys), 10)
    eq_(max(len(p) for p in decoys), 15)

def test_generate_decoy_sequence_groups():
    groups = generate_decoy_sequence_groups(n_decoy_loci=10, decoys_per_locus=3)
    eq_(len(groups), 10)
    for group in groups:
        eq_(len(group.children), 3)
        eq_(len(group.binding_cores), 1)