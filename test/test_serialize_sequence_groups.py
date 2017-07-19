from mhc2.sequence_group import (
    serialize_sequence_groups_to_json,
    deserialize_sequence_groups_from_json,
    SequenceGroup,
)
from nose.tools import eq_

def test_sequence_group_serialization():
    sequence_group = SequenceGroup(
        contig="A" * 12 + "CSCSQLYFM",
        leaves={"CSCSQLYFM"},
        binding_cores={"CSCSQLYFM"},
        children=[
            "A" * 10 + "CSCSQLYFM",
            "A" * 12 + "CSCSQLYFM",
            "A" * 3 + "CSCSQLYFM"])
    sequence_groups = [sequence_group]
    json = serialize_sequence_groups_to_json(sequence_groups)
    reconstructed = deserialize_sequence_groups_from_json(json)
    eq_(sequence_groups, reconstructed)
