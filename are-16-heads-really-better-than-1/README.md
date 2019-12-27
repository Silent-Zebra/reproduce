# Are Sixteen Heads Really Better than One?

This repository contains code to reproduce the experiments in our paper [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650).

## Prerequisite

First, you will need python >=3.6 with `pytorch>=1.0`. Then, clone our forks of `fairseq` (for MT experiments) and `pytorch-pretrained-BERT` (for BERT):

```bash
# Fairseq
git clone https://github.com/pmichel31415/fairseq
# Pytorch pretrained BERT
git clone https://github.com/pmichel31415/pytorch-pretrained-BERT
cd pytorch-pretrained-BERT
git checkout paul
python setup.py install
cd ..
```

You will also need `sacrebleu` to evaluate BLEU score  (`pip install sacrebleu`).

## Ablation experiments

### get GLUE data
```bash
git clone https://github.com/nyu-mll/GLUE-baselines.git
cd GLUE-baselines
python download_glue_data.py
mv glue_data ..
cd ..
```

Extra perparation of GLUE data: MNLI-Mistmatch
```bash
cd glue_data
mkdir mnli-mis
cd MNLI
mv dev_mismatched.tsv ../mnli-mis
mv test_mismatched.tsv ../mnli-mis
mv train.tsv ../mnli-mis
```

### BERT

Running

```bash
bash experiments/BERT/heads_ablation.sh MNLI
```

To perform head ablation on MNLI mismatched
```
bash experiments/BERT/heads_ablation.sh mnli-mis
```

Will fine-tune a pretrained BERT on MNLI (stored in `./models/MNLI`) and perform the individual head ablation experiment from Section 3.1 in the paper alternatively you can run the experiment with `CoLA`, `MRCP` or `SST-2` as a task in place of `MNLI`.

### MT

You can obtain the pretrained WMT model from [this link from the fairseq repo](wget https://s3.amazonaws.com/fairseq-py/models/wmt14.en-fr.joined-dict.transformer.tar.bz2). Use the [Moses tokenizer](https://github.com/moses-smt/mosesdecoder) and [subword-nmt](https://github.com/rsennrich/subword-nmt) in conjunction to the BPE codes provided with the pretrained model to prepair any input file you want. Then run:

```bash
bash experiments/MT/wmt_ablation.sh $BPE_SEGMENTED_SRC_FILE $DETOKENIZED_REF_FILE
```

## Systematic Pruning Experiments

### BERT

To iteratively prune 10% heads in order of increasing importance run

```bash
bash experiments/BERT/heads_pruning.sh MNLI --normalize_pruning_by_layer
```

This will reuse the BERT model fine-tuned if you have run the ablation experiment before (otherwise it'll just fine-tune it for you). The output of this is **very** verbose, but you can get the gist of the result by calling `grep "strategy\|results" -A1` on the output.

### WMT

Similarly, just run:

```bash
bash experiments/MT/prune_wmt.sh $BPE_SEGMENTED_SRC_FILE $DETOKENIZED_REF_FILE
```

You might want to change the paths in the experiment files to point to the binarized fairseq dataset on whic you want to estimate importance scores.

## Graphing

Modify the graphing script accordingly to filepath you saved the results

Graph in Section 3.2
filenames are created by task_model_description, if model part of the filename is missing, then it implies that the script can be used for both BERT and IWSLT/WMT

```bash
python task3-2_bert_histogram.py
python task3-2_wmt_histogram.py
```
output from task3-2_bert_histogram.py also contains index of the values that has statistical significance. The index//12 is the row and index%12 is the column 

Graph in Section 3.3
```bash
python task3-3_bert_all-but-one.py
```
Table for task 4 can be done in excel with given statistical significance file and head ablation all but one file


