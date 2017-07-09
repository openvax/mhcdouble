# MHCdouble

Predicting presentation of peptides by Class II MHC molecules using convolutional neural networks.

This library allows you to train and predict with a collection of allele-specific MHC II
peptide presentation models from categorical binding data (such as from a
mass spec. assay).

## Training Convolutional Predictors

Here's an example of a minimal invocation required to train a model on a new
dataset:

```
mhc2-train-convolutional \
    --model-dir trained-models \
    --dataset dataset.csv
```

The file `dataset.csv` must contain the columns `allele`, `peptide`, and `hit`.


You can also specifiy training parameters and architectural hyperparameters:

```
mhc2-train-convolutional \
    --model-dir trained-models \
    --dataset dataset.csv \
    --conv-activation linear \
    --global-batch-normalization 0 \
    --embedding-dim 10 \
    --conv-dropout 0 \
    --batch-size 16 \
    --patience 2
```

A more complete list of parameters can be seen by running `mhc2-train --help`.

## Prediction

The following code will generate binding predictions for one or more alleles on a given set of peptides:

```
mhc2-predict \
    --model-dir trained-models \
    --allele DRB1*01:01 \
    --peptides-file peptides.txt
```

Additional alleles can be specified as part of the `--allele` flag, such as `--allele DRB1*01:01 DRB1*11:01`.

Peptides can also be given manually on the commandline using the `--peptide` option, such as `--peptide SIINFEKLQQQQQQ`.

If you want to make predictions for different peptides associated with each allele, you can provide a CSV file with columns "allele" and "peptide" using the `--input-csv` option.

## Which alleles are available?

To see which alleles are available in a directory of trained models, run:

```mhc2-list-alleles MODEL-DIR```
