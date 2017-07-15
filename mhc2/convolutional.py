from pepnet import Predictor, SequenceInput, Output
from ..ensemble import Ensemble

FIRST_CONV_SIZES = {1: 4, 3: 4, 8: 4, 9: 16, 10: 4}
SECOND_CONV_SIZES = {1: 8, 3: 8, 5: 8, 7: 8, 9: 8}
CONV_DROPOUT = 0.25
CONV_ACTIVATION = "relu"
DENSE_DROPOUT = 0.25
DENSE_ACTIVATION = "relu"
GLOBAL_BATCH_NORMALIZATION = True

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
