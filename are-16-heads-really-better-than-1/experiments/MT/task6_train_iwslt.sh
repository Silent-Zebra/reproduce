#!/usr/bin/env bash

# Data/model folders
TEXT_ROOT=iwslt14.tokenized.de-en
DATA_BIN=data-bin/iwslt14.tokenized.de-en
#CKPT_ROOT=./projects/tir3/users/pmichel1/checkpoints 
CKPT_ROOT=./task6_checkpoints
TRAIN=${TRAIN:-1}


# Produces a hash from an experiment name, used to get fixed random seed
# Courtesy of https://stackoverflow.com/a/7265130
function get_seed (){
  # Hash
  n=$(md5sum <<< "$1")
  # Convert to decimals
  n=$((0x${n%% *}))
  # Take the absolute value
  n=${n#-}
  # Modulo 2^32 - 1 for fairseq
  n=$(( n % 4294967295 ))
  # + 1 so that the seed is not 0
  echo $(( n + 1 ))
}

EXP_NAME="iwslt14_de-en_8head_before_${SLURM_ARRAY_TASK_ID}"
SEED=`get_seed $EXP_NAME`
TRAIN_OPTIONS=${--seed $SEED --arch transformer_iwslt_de_en_8head_before:2}

CKPT_DIR=$CKPT_ROOT/$EXP_NAME

# bash fairseq/examples/translation/prepare-iwslt14.sh


if [ $TRAIN == 1 ]
then
    echo 'Training'
    mkdir -p $CKPT_DIR
    python fairseq/task6_train.py \
        $DATA_BIN \
        -s de \
        -t en \
        --seed $SEED \
        --arch 'fconv' \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)'\
        --lr 0.0005 \
        --warmup-updates 4000 \
        --warmup-init-lr '1e-07' \
        --min-lr '1e-09' \
        --label-smoothing 0.1 \
        --dropout 0.3 \
        --max-tokens 4000 \
        --lr-scheduler inverse_sqrt \
        --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy \
        --max-update 50000 \
        --save-dir $CKPT_DIR \
		--no-progress-bar \
		--beam 5 --lenpen 1 --remove-bpe "@@ " \
		--normalize-by-layer \
		--batch-size 64 \
		--reset-optimizer
fi
