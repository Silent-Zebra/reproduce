#!/usr/bin/env bash

TASK=$1
OPTIONS="${@:2}"

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $here/prepare_task.sh $TASK


echo -n "Base acc: "
echo $base_acc
echo -n "Part: "
echo $part
for layer in `seq 1 12`
do
    echo -n "Layer: "
    echo -n "$layer"
    for head in `seq 1 12`
    do
        mask_str="${layer}:"
		for subhead in `seq 1 12`
		do
			if [[ ${subhead} != ${head} ]]
			then
				mask_str+="$subhead,"
			fi
		done
        acc=$(run_eval "--attention_mask_heads $mask_str $OPTIONS" | grep $metric | rev | cut -d" " -f1 | rev)
        printf "\t%.5f" $(echo "$acc - $base_acc" | bc )
    done
done

