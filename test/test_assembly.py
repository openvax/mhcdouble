from mhc2.assembly import naive_left_overlap, find_overlaps

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
