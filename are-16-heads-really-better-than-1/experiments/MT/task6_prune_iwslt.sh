#!/usr/bin/env bash
DATA_BIN=data-bin/iwslt14.tokenized.de-en
ARCH=${2:-"transformer_iwslt_de_en_8head_before"}
EXTRA_OPTIONS=$3

epochlist=(1 2 3 5 7 10 15 20 25 30 35 40)

for i in ${epochlist[@]}
do
    MODEL=iwslt14_de-en_8head_before_/checkpoint${i}.pt 
    echo $MODEL
    python fairseq/prune.py \
        $DATA_BIN \
        -s de \
        -t en \
        --restore-file $MODEL \
        --arch $ARCH \
        --normalize-by-layer \
        --reset-optimizer \
        --batch-size 64 \
        --reset-optimizer \
        --beam 5 --lenpen 1 --remove-bpe "@@ " --raw-text $EXTRA_OPTIONS \
        --no-progress-bar | grep BLEU
done



