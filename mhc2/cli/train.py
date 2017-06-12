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

from argparse import ArgumentParser
from collections import defaultdict

from pepnet import Predictor, SequenceInput, Output

from .common import parse_args, parse_bool
from ..dataset import Dataset
from ..model_collection import ModelCollection
from ..ensemble import Ensemble
from ..training import train_model_collection

N_CV_SPLITS = 3
MAX_EPOCHS = 200
PATIENCE = 3

FIRST_CONV_SIZES = {1: 4, 3: 4, 8: 4, 9: 16, 10: 4}
SECOND_CONV_SIZES = {1: 8, 3: 8, 5: 8, 7: 8, 9: 8}
GLOBAL_BATCH_NORMALIZATION = True
CONV_DROPOUT = 0.25
CONV_ACTIVATION = "relu"
DENSE_DROPOUT = 0.25
DENSE_ACTIVATION = "relu"

parser = ArgumentParser(description="Predict peptide-MHC Class II binding")

parser.add_argument(
    "--model-dir",
    required=True,
    help="Directory to save trained MHC II models")

parser.add_argument(
    "--dataset",
    help=("""CSV with the following columns:
    - allele : string
    - peptide : string
    - hit : binary
        1 = peptide is a mass-spec hit
        0 = peptide is a decoy
    - weight : float (optional)
    - group_id : int (optional)

    Can be generated from just hits using `mhc2-generate-decoys`
    and `mhc2-group-nested-seqs`."""))


parser.add_argument("--log-filename", help="Optional CSV log of models trained")

hyperparam_parser = parser.add_argument_group("Model Hyperparameters")
hyperparam_parser.add_argument(
    "--max-peptide-length",
    help="Maximum length of input peptide, inferred from training data if omitted")

hyperparam_parser.add_argument(
    "--global-batch-normalization",
    default=GLOBAL_BATCH_NORMALIZATION,
    type=parse_bool,
    help="Apply BatchNorm to globally pooled activations, '0' or '1'")

hyperparam_parser.add_argument(
    "--embedding-dim",
    type=int,
    default=24,
    help="Number of dimensions for amino acid embedding")

hyperparam_parser.add_argument(
    "--conv-dropout",
    default=0.25,
    type=float,
    help="Fraction of convolutional activations to randomly drop during training")

hyperparam_parser.add_argument(
    "--conv-activation",
    default="relu",
    help="Nonlinearity applied to convolutional layers")

training_parser = parser.add_argument_group("Training")
training_parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
training_parser.add_argument("--patience", type=int, default=PATIENCE)
training_parser.add_argument("--cross-validation-splits", type=int, default=3)
training_parser.add_argument("--batch-size", type=int, default=64)

def make_predictor(
        max_peptide_length,
        add_start_tokens=True,
        add_stop_tokens=True,
        embedding_dim=24,
        conv_filter_sizes=[FIRST_CONV_SIZES, SECOND_CONV_SIZES],
        conv_dropout=CONV_DROPOUT,
        conv_activation="relu",
        global_pooling_batch_normalization=GLOBAL_BATCH_NORMALIZATION,
        dense_layer_sizes=[],
        dense_dropout=DENSE_DROPOUT,
        dense_activation=DENSE_ACTIVATION):
    return Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=max_peptide_length,
                add_start_tokens=add_start_tokens,
                add_stop_tokens=add_stop_tokens,
                variable_length=True,
                embedding_dim=embedding_dim,
                conv_filter_sizes=conv_filter_sizes,
                conv_activation=conv_activation,
                conv_dropout=conv_dropout,
                global_pooling=True,
                global_pooling_batch_normalization=global_pooling_batch_normalization),
            outputs=Output(
                1,
                activation="sigmoid",
                dense_layer_sizes=dense_layer_sizes,
                dense_activation=dense_activation,
                dense_dropout=dense_dropout))


def make_predictor_from_args(max_peptide_length, args):
    return make_predictor(
        max_peptide_length=max_peptide_length,
        embedding_dim=args.embedding_dim,
        conv_dropout=args.conv_dropout,
        conv_activation=args.conv_activation,
        global_pooling_batch_normalization=args.global_batch_normalization)


def main(args_list=None):
    args = parse_args(parser=parser, args_list=args_list)
    print("Args: %s" % (args,))
    dataset = Dataset.from_csv(args.dataset)

    max_peptide_length = args.max_peptide_length
    if not max_peptide_length:
        max_peptide_length = max(len(p) for p in dataset.peptides)

    model_collection = train_model_collection(
        path=args.model_dir,
        dataset=dataset,
        make_predictor_fn=lambda: make_predictor_from_args(max_peptide_length, args),
        n_cv_splits=args.cross_validation_splits,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        csv_log_filename=args.log_filename)
    model_collection.to_disk()


