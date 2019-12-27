#!/usr/bin/env bash

DATA_BIN=wmt14_en-fr/wmt14.en-fr.joined-dict.newstest2014
MODEL=wmt14_en-fr/wmt14.en-fr.joined-dict.transformer/model.pt
EXTRA_OPTIONS=$1

python fairseq/prune.py \
    $DATA_BIN \
    -s en \
    -t fr \
    -a transformer_vaswani_wmt_en_fr_big \
    --restore-file $MODEL \
    --share-all-embeddings \
    --normalize-by-layer \
    --reset-optimizer \
    --batch-size 16 \
    --reset-optimizer \
    --beam 5 --lenpen 1 --remove-bpe "@@ " --raw-text $EXTRA_OPTIONS \
    --no-progress-bar

