from mhc2.assembly import (
    naive_left_overlap,
    find_overlaps,
    assemble_into_sequence_groups
)

from nose.tools import eq_


def test_naive_left_overlap():
    s1 = "AAABBBCCC"
    s2 = "BBBCCCDDD"
    eq_(naive_left_overlap(s1, s2), 6)
    eq_(naive_left_overlap(s2, s1), 0)

def test_find_overlaps():
    s1 = "AAABBBCCC"
    s2 = "BBBCCCDDD"
    eq_(dict(find_overlaps([s1, s2], min_overlap_size=6)), {s1: [s2]})
    eq_(dict(find_overlaps([s1, s2], min_overlap_size=7)), {})

def test_assemble_into_sequence_groups():
    s1 = "AAABBBCCC"
    s2 = "BBBCCCDDD"
    groups = assemble_into_sequence_groups([s1, s2], min_overlap_size=6)
    eq_(len(groups), 1)
    g = groups[0]
    eq_(g.contig, "AAABBBCCCDDD")
