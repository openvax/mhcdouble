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
from itertools import chain, combinations
import numpy as np

def groupby(xs, keys):
    result = defaultdict(list)
    for (x, key) in zip(xs, keys):
        result[key].append(x)
    return result

def powerset(iterable):
    """
    Powerset implementation from
    https://docs.python.org/2/library/itertools.html#itertools.combinations
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def all_splits(items, nonempty=False):
    """
    Split given items into all possible subsets (s1, s2) such that
    s1.union(s2) gives back the original items.
    """
    full_set = set(items)
    for s1 in powerset(full_set):
        s1 = set(s1)
        s2 = full_set.difference(s1)
        if nonempty and (len(s1) == 0 or len(s2) == 0):
            continue
        yield s1, s2

def shuffled_list(lst):
    n = len(lst)
    indices = np.arange(n)
    np.random.shuffle(indices)
    return [lst[i] for i in indices]

