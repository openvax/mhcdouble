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

import pandas as pd
import numpy as np
import ujson

class Ensemble(object):
    """
    Ensemble of allele-specific Class II MHC binding predictors
    which can be serialized, trained, and used to make predictions.
    """
    def __init__(self, models=[], model_weights=None):
        self.models = models
        if model_weights is None:
            model_weights = np.ones(len(self.models), dtype="float32")
        if len(model_weights) != len(self.models):
            raise ValueError(
                "Mismatch between %d models and %d weights" % (
                    len(self.models), len(self.model_weights)))
        self.model_weights = model_weights

    def add_model(self, model, weight=1.0):
        self.models.append(model)
        self.model_weights = np.array(list(self.model_weights) + [weight])

    def predict(self, peptides):
        total_pred = np.zeros(len(peptides), dtype="float64")
        total_weight = 0.0
        for model, weight in zip(self.models, self.model_weights):
            pred = model.predict(peptides)
            total_pred += pred
            total_weight += weight
        return total_pred / total_weight

    def predict_dataframe(self, peptides):
        pred = self.predict(peptides)
        return pd.DataFrame({"peptide": peptides, "pred": pred})

    def to_dict(self):
        result = {"model_weights": self.model_weights}
        model_dicts = []
        for model in self.models:
            model_dicts.append(model.to_dict())
        result["models"] = model_dicts
        return result

    @classmethod
    def from_dict(cls, d):
        # importing locally to avoid adding a significant delay to
        # the import of every module
        from pepnet import Predictor

        model_dicts = d["models"]
        model_weights = d.get("model_weights")
        models = [
            Predictor.from_dict(d)
            for d in model_dicts
        ]
        return cls(models=models, model_weights=model_weights)

    def to_json(self):
        return ujson.dumps(self.to_dict())

    def to_json_file(self, filename):
        with open(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_string):
        d = ujson.loads(json_string)
        if len(d) == 0:
            raise ValueError("Dictionary representation of ensemble empty")
        return cls.from_dict(d)

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "r") as f:
            json_string = f.read().strip()
        if len(json_string) == 0:
            raise ValueError("JSON file '%s' was empty" % (filename,))
        return cls.from_json(json_string)
