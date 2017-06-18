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

"""
Save different training runs by name and load them from disk.
"""

import os
import os.path
from shutil import rmtree

from .mhc_names import normalize_mhc_name
from .ensemble import Ensemble

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ModelCollection(object):
    """
    Collection of allele-specific ensembles which can give the names
    of supported alleles without loading them.
    """
    def __init__(self, path):
        self._path = path
        self._allele_to_ensemble_dict = {}

    def path(self, create_if_missing=False):
        if create_if_missing:
            _ensure_dir(self._path)
        return self._path

    def exists(self):
        return os.path.exists(self.path(create_if_missing=False))

    def delete(self):
        if self.exists():
            rmtree(self.path())

    def clear(self):
        self._allele_to_ensemble_dict.clear()

    def allele_to_path_dict(self):
        result = {}
        for filename in os.listdir(self.path(create_if_missing=True)):
            if not filename.endswith(".json"):
                continue
            if filename.startswith("_") or filename.startswith("."):
                continue
            allele = self._filename_to_allele(filename)
            path = os.path.join(self.path(), filename)
            result[allele] = path
        return result

    def alleles(self):
       return sorted(set(self.allele_to_path_dict().keys()))

    def alleles_to_ensembles(self):
        for allele, path in self.allele_to_path_dict().items():
            if allele not in self._allele_to_ensemble_dict:
                ensemble = Ensemble.from_json_file(path)
                self._allele_to_ensemble_dict[allele] = ensemble
        return self._allele_to_ensemble_dict

    def __getitem__(self, allele):
        allele_to_path_dict = self.allele_to_path_dict()
        allele = normalize_mhc_name(allele)
        if allele not in allele_to_path_dict:
            print("Available alleles:")
            for other_allele in self.alleles():
                print("-- %s" % other_allele)
            raise KeyError("Allele not found: %s" % allele)
        elif allele not in self._allele_to_ensemble_dict:
            path = allele_to_path_dict[allele]
            ensemble = Ensemble.from_json_file(path)
            self._allele_to_ensemble_dict[allele] = ensemble
        return self._allele_to_ensemble_dict[allele]

    def _allele_to_filename(self, allele):
        basename = allele.replace("*", "_")
        return basename + ".json"

    def _filename_to_allele(self, filename):
            basename = os.path.basename(filename)
            without_extension = os.path.splitext(basename)[0]
            return normalize_mhc_name(without_extension)

    def _allele_to_path(self, allele, create_if_missing=True):
        filename = self._allele_to_filename(allele)
        return os.path.join(
            self.path(create_if_missing=create_if_missing), filename)

    def add_single_model(self, allele, model, weight=1.0):
        if allele in self._allele_to_ensemble_dict:
            self.alleles_to_ensembles()[allele].add_model(model, weight)
        else:
            self.alleles_to_ensembles()[allele] = Ensemble(
                models=[model],
                model_weights=[weight])

    def add_ensemble(self, allele, ensemble):
        self.alleles_to_ensembles()[allele] = ensemble

    def to_disk(self):
        for allele, ensemble in self.alleles_to_ensembles().items():
            path = self._allele_to_path(allele, create_if_missing=True)
            print("-- Writing %s" % path)
            json_string = ensemble.to_json()
            with open(path, "w") as f:
                f.write(json_string)
