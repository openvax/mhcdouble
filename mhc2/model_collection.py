"""
Save different training runs by name and load them from disk.
"""

import os
import os.path
from shutil import rmtree

from .mhc_names import normalize_mhc_name

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ModelCollection(object):
    def __init__(self, path):
        self._path = path
        self._cache = {}

    def clear_cache(self):
        self.cache.clear()

    def path(self, create_if_missing=False):
        if create_if_missing:
            _ensure_dir(self._path)
        return self._path

    def exists(self):
        return os.path.exists(self.path(create_if_missing=False))

    def delete(self):
        if self.exists():
            rmtree(self.path())

    def _allele_to_filename(self, allele):
        basename = allele.replace("*", "_")
        return basename + ".json"

    def _filename_to_allele(self, filename):
            basename = os.path.basename(filename)
            without_extension = os.path.splitext(basename)[0]
            return normalize_mhc_name(without_extension)

    def alleles(self):
        allele_names = []
        for filename in os.listdir(self.path(create_if_missing=True)):
            if not filename.endswith(".json"):
                continue
            if filename.startswith("_") or filename.startswith("."):
                continue
            allele = self._filename_to_allele(filename)
            allele_names.append(allele)
        return allele_names

    def _allele_path(self, allele):
        filename = self._allele_to_filename(allele)
        return os.path.join(
            self.path(create_if_missing=True), filename)

    def add_model(self, allele, model):
        allele = parse_mhc_allele(allele)
        self._cache[allele] = model
        path = self._allele_path(allele)
        json_string = model.to_json()
        with open(path, "w") as f:
            w.write(json_string)
