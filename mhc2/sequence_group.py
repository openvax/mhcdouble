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
import ujson

from serializable import Serializable

class SequenceGroup(Serializable):
    def __init__(
            self,
            contig,
            children,
            leaves,
            binding_cores=[]):
        self.contig = contig
        self.children = list(children)
        self.leaves = list(set(leaves))
        self.binding_cores = list(set(binding_cores))

    def to_dict(self):
        return {
            "contig": self.contig,
            "children": self.children,
            "leaves": self.leaves,
            "binding_cores": self.binding_cores
        }

    def to_json(self):
        return ujson.dumps(self.to_dict())

    def from_json(self, json_string):
        return self.from_dict(ujson.loads(json_string))

    @classmethod
    def from_dict(self, d):
        return SequenceGroup(**d)

def flatten_sequence_groups(sequence_groups):
    """
    Returns list of children and list of group_ids for each child sequence
    """
    child_sequences = []
    group_ids = []
    contigs = []
    binding_cores = []
    for group_id, group in enumerate(sequence_groups):
        for c in group.children:
            child_sequences.append(c)
            group_ids.append(group_id)
            contigs.append(group.contig)
            binding_core = ""
            for seq in group.binding_cores:
                if seq in c:
                    binding_core = seq
                    break
            binding_cores.append(binding_core)
    return child_sequences, group_ids, contigs, binding_cores

def print_sequence_group_stats(groups):
    print("# groups = %d" % len(groups))
    children_per_group = [len(g.children) for g in groups]
    print("# sequences per group min = %d, mean = %f, max = %d (%s)" % (
        min(children_per_group),
        np.mean(children_per_group),
        max(children_per_group),
        max(groups, key=lambda g: len(g.children) == max(children_per_group)).contig))
    leaves_per_group = [len(g.leaves) for g in groups]
    print("# leaves per group min = %d, mean = %f, max = %d (%s)" % (
        min(leaves_per_group),
        np.mean(leaves_per_group),
        max(leaves_per_group),
        max(groups, key=lambda g: len(g.leaves) == max(leaves_per_group)).contig))
    binding_cores_per_group = [len(g.binding_cores) for g in groups]
    print("# binding cores per group min = %d, mean = %f, max = %d (%s)" % (
        min(binding_cores_per_group),
        np.mean(binding_cores_per_group),
        max(binding_cores_per_group),
        max(groups, key=lambda g: len(g.binding_cores) == max(binding_cores_per_group)).contig))
    print("# of groups with more than one binding core: %d" % (
        sum(x > 1 for x in binding_cores_per_group)))
    child_to_binding_core_ratios = [len(g.children) / len(g.binding_cores)  for g in groups]
    print("Ratio of children to binding cores min = %f, mean = %f, max = %f" % (
        np.min(child_to_binding_core_ratios),
        np.mean(child_to_binding_core_ratios),
        np.max(child_to_binding_core_ratios)))
    binding_core_sequences = []
    for g in groups:
        binding_core_sequences.extend(g.binding_cores)
    binding_core_lengths = [len(c) for c in binding_core_sequences]
    print("Binding core length min = %d, mean = %0.4f, max = %d" % (
        min(binding_core_lengths),
        np.mean(binding_core_lengths),
        max(binding_core_lengths)))

def save_sequence_groups_to_txt_file(groups, path):
    with open(path, "w") as f:
        for i, g in enumerate(groups):
            f.write("%d) contig length = %d, # hits = %d, # binding cores = %d\n" % (
                i + 1, len(g.contig), len(g.children), len(g.binding_cores)))
            label = "Assembled sequence: "
            f.write(label + "%s\n" % g.contig)
            if len(g.binding_cores) == 1 and len(g.binding_cores[0]) == len(g.contig):
                continue
            for c in g.children:
                n_spaces = g.contig.index(c)
                f.write(" " * len(label) + " " * n_spaces + c + "\n")

def serialize_sequence_groups_to_json(groups):
    group_json_strings = [g.to_json() for g in groups]
    return "[\n%s\n]\n" % (",\n".join(group_json_strings))

def save_sequence_groups_to_json_file(groups, path):
    with open(path, "w") as f:
        f.write(serialize_sequence_groups_to_json(groups))

def deserialize_sequence_groups_from_json(json_string):
    list_of_dict = ujson.loads(json_string)
    return [SequenceGroup.from_dict(d) for d in list_of_dict]

def load_sequence_groups_from_json_file(path):
    with open(path, 'r') as f:
        return deserialize_sequence_groups_from_json(f.read())
