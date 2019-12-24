#!/usr/bin/env bash

# source env/bin/activate

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DATA_BIN=./wmt14_en-fr/wmt14.en-fr.joined-dict.newstest2014
MODEL=./wmt14_en-fr/wmt14.en-fr.joined-dict.transformer/model.pt
# DATA_BIN=/projects/tir3/users/pmichel1/data-bin/wmt14.en-fr.joined-dict.newstest2013
# MODEL=/projects/tir3/users/pmichel1/checkpoints/wmt14.en-fr.joined-dict.transformer/model.pt
MOSES_SCRIPTS="fairseq/examples/translation/mosesdecoder/scripts"
OUT_DIR=output
# tokenized en
SRC_FILE="wmt14_en_fr_data/newstest2014.tok.en"
REF_FILE="wmt14_en_fr_data/newstest2014.fr"

OUT_PREFIX="newstest2014_en-fr"
# Use the following instead for ablating all but one head in a layer
# OUT_PREFIX=newstest2014_en-fr.allbut

# Iterate over the 3 "parts" of the model, Enc-Enc (E), Enc-Dec (A) and Dec-Dec (D)
for part in "E" "A" "D"
do
    echo $part
    for layer in `seq 1 6`
    do
        echo -n "$layer"
        for head in `seq 1 16`
        do
            mask_str="${part}:${layer}:${head}"
            pval=$(cat $OUT_DIR/${OUT_PREFIX}.${mask_str}.out.fr | grep p= | cut -d "=" -f2)
            printf "\t%.3f" $(echo $pval)
        done
        echo ""
    done
done
