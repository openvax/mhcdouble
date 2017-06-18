# mhc2
Class II MHC binding and antigen processing prediction

This library allows you to train and predict with a collection of allele-specific MHC II
peptide presentation models from categorical binding data (such as from a
mass spec. assay).

## Training

Here's an example of a minimal invocation required to train a model on a new
dataset:

```
mhc2-train \
    --model-dir trained-models \
    --dataset dataset.csv
```

The file `dataset.csv` must contain the columns `allele`, `peptide`, and `hit`.


You can also specifiy training parameters and architectural hyperparameters:

```
mhc2-train \
    --model-dir trained-models \
    --dataset dataset.csv \
    --conv-activation linear \
    --global-batch-normalization 0 \
    --embedding-dim 10 \
    --conv-dropout 0 \
    --batch-size 16 \
    --patience 2
```

## Prediction


```
mhc2-predict \
    --model-dir trained-models \
    --allele DRB1*01:01 \
    --peptides-file peptides.txt
```