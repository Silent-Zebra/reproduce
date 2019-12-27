# Reproducing "Are Sixteen Heads Really Better than One?"

This repository contains the code we used to reproduce the results of [Michel et al., 2019, "Are Sixteen Heads Really Better than One?"](https://arxiv.org/abs/1905.10650). We used the [authors' original repository](https://github.com/pmichel31415/are-16-heads-really-better-than-1) as a starting point, and made modifications and extensions to the code.

## Prerequisites

You will need python >=3.6 with `pytorch>=1.0`. We have already included in this repository modifications of the authors' clones of `fairseq` (for MT experiments) and `pytorch-pretrained-BERT` (for BERT).

You will need `sacrebleu` to evaluate BLEU score  (`pip install sacrebleu`).

## Experiments

We organize documentation below based on what we used for each section of the paper. All commands we run for experiments start from the base folder [are-16-heads-really-better-than-1](https://github.com/Silent-Zebra/reproduce/tree/master/are-16-heads-really-better-than-1/)

### 3.2 - Ablating One Head

#### BERT

For BERT, run 

```bash
bash experiments/BERT/heads_ablation.sh MNLI
```

to fine-tune a pretrained BERT on MNLI (stored in `./models/MNLI`) and perform the individual head ablation experiment. The experiment can also be run with `CoLA` or `SST-2` as tasks in place of `MNLI`. These datasets can be obtained from the [GLUE Baselines repository](https://github.com/nyu-mll/GLUE-baselines) which we have already included in this repository as well.

To get GLUE data:
```bash
cd GLUE-baselines
python download_glue_data.py
mv glue_data ..
cd ..
```

Extra perparation of GLUE data for the MNLI-Mismatched dataset used in section 3.4:
```bash
cd glue_data
mkdir mnli-mis
cd MNLI
mv dev_mismatched.tsv ../mnli-mis
mv test_mismatched.tsv ../mnli-mis
mv train.tsv ../mnli-mis
```

Statistical significance testing is included in the task3-2_bert_histogram.py file.

#### WMT and IWSLT

We got the models and DATA_BIN files from [here](https://github.com/pytorch/fairseq/tree/master/examples/translation).

We obtained tokenized data from [here](https://github.com/google/seq2seq/blob/master/docs/nmt.md).

We then modified the script taken from [here](https://github.com/google/seq2seq/blob/master/bin/data/wmt16_en_de.sh).

To set up IWSLT and WMT, follow the instructions [here](https://github.com/Silent-Zebra/reproduce/tree/master/are-16-heads-really-better-than-1/fairseq/examples/translation)

For WMT, run

```bash
bash experiments/MT/wmt_ablation.sh
```

For statistical significance testing, run

```bash
bash experiments/MT/stat_sig_test_wmt.sh
```

For IWSLT, run

```bash
bash experiments/MT/iwslt_ablation.sh
```

By default, we use English-to-French test sets for WMT and German-to-English for IWSLT. You can modify the file paths in the sh files mentioned above to point to the desired datasets.


### 3.3 - Ablating All But One Head

The procedure is the same as above, but you need to modify the sh files (replace the lines as described in the files).

### 3.4 - Correlation Among Heads

Use task3-4_correlation.py, applied to the output of 3.2.

### 4 - Systematic Pruning Experiments

#### BERT

To iteratively prune 5% of heads in order of increasing importance run

```bash
bash experiments/BERT/heads_pruning.sh MNLI --normalize_pruning_by_layer
```

This will reuse the BERT model fine-tuned if you have run the ablation experiment before (otherwise it'll just fine-tune it for you). The output of this is **very** verbose, but you can get the gist of the result by calling `grep "strategy\|results" -A1` on the output.

To prune by a static evaluation of accuracy from ablating each head individually, we first ran the experiment in 3.2, and saved the output to a CSV file (automatically done when using our code). Then run:

```bash
bash experiments/BERT/heads_pruning.sh MNLI --prune_by_accuracy=True --prune_by_accuracy_file=32BERT_test.csv
```

#### IWSLT

To iteratively prune 10% of heads in order of increasing importance run:

```bash
bash experiments/MT/prune_iwslt.sh iwslt14_de-en_8head_before_/checkpoint_last.pt 
```

changing iwslt14_de-en_8head_before_/checkpoint_last.pt to point to whichever model checkpoint you would like to load (checkpoints are generated from the preparation/training steps mentioned in 3.2 above, when setting up IWSLT).

We used the experiments/MT/prune_iwslt_acc.sh script instead of the above to prune by accuracy. This assumes there exists a file called "iwslt_ablation_out_notext.txt". An example file with this format can be found in our results/ folder. fairseq/prune_acc.py can be modified to make this process more dynamic.

### 5 - When Are More Heads Important? The Case of Machine Translation

We ran the following commands:

```bash
bash experiments/MT/prune_iwslt.sh iwslt14_de-en_8head_before_/checkpoint_last.pt transformer_iwslt_de_en_8head_before --encoder-self-only 
bash experiments/MT/prune_iwslt.sh iwslt14_de-en_8head_before_/checkpoint_last.pt transformer_iwslt_de_en_8head_before --encoder-decoder-only 
bash experiments/MT/prune_iwslt.sh iwslt14_de-en_8head_before_/checkpoint_last.pt transformer_iwslt_de_en_8head_before --decoder-self-only 
```

where again, the argument "iwslt14_de-en_8head_before_/checkpoint_last.pt" points to the file location of the model checkpoint.

### 6 - Dynamics of Head Importance during Training

Run

```bash
bash experiments/MT/task6_prune_iwslt.sh 
```
The script assumes that checkpoints are in the folder iwslt14_de-en_8head_before_. This can be changed in the script.


## Graphing

Filenames are created in the format of $task_$model_$description. If the model part of the filename is missing, then it means that the script can be used for both BERT and IWSLT/WMT.
Modify the graphing script according to filepath where you saved the results, and the model used for the experiment if applicable


### Graphs in Section 3.2

Run

```bash
python task3-2_bert_histogram.py
python task3-2_wmt_histogram.py
```
The output from task3-2_bert_histogram.py also contains the index of the values that have statistical significance. The index//12 is the row and index%12 is the column 

### Graphs in Section 3.3

Run

```bash
python task3-3_bert_all-but-one.py
```

Table for task 4 can be done in excel with the given statistical significance file and the head ablation for all but one file

### Graph in Section 3.4

Run

```bash
python task3-4_correlation.py
```
This script can produce graphs for both BERT and WMT/IWSLT by changing model variable in the script.

### Graph in Section 4

Run

```bash
python task4_generate_trend.py
```

### Graph in Section 5

Run

```bash
python task5_iwslt_separate-prune.py
```

### Graph in Section 6

Run

```bash
python task6_iwslt_training-dynamic.py
```