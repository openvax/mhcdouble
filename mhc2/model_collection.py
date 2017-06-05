"""
Save different training runs by name and load them from disk.
"""

import os
import os.path
from shutil import rmtree

from appdirs import user_data_dir

BASE_PATH = user_data_dir(
    appname="mhc2",
    appauthor="hammerlab",
    version="1")

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ModelCollection(object):
    def __init__(self, name):
        self.name = name
        self._cache = {}

    def clear_cache(self):
        self.cache.clear()

    def path(self, create_if_missing=False):
        path = os.path.join(BASE_PATH, self.name)
        if create_if_missing:
            _ensure_dir(path)
        return path

    def exists(self):
        return os.path.exists(self.path(create_if_missing=False))

    def delete(self):
        if self.exists():
            rmtree(self.path())

    def alleles(self):
        raw_allele_names = []
        for filename in os.listdir(self.path(create_if_missing=True)):
            if not filename.startswith("_") or filename.startswith("."):
                raw_allele_names.append(filename)
        return [parse_mhc_alleles(allele) for allele in raw_allele_names]

    def allele_dir(self, allele, create_if_missing=False):
        path = self.path(create_if_missing=create_if_missing)
        path = os.path.join(path, allele)
        if create_if_missing:
            _ensure_dir(path)
        return path

    def add_model(self, allele, model):
        self._cache[allele] = model
        json_string = model.to_json()
        directory = self.allele_dir(allele)
        allele_json_name = allele + ".json"
        path = os.path.join(directory, allele_json_name)
        with open(path, "w") as f:
            w.write(json_string)

