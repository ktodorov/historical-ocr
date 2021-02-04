#!/bin/bash
#SBATCH --job-name=ft-ocr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=60000M
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1

module purge
module load 2019

module load Miniconda3
source activate historical-ocr

CONF="$CONFMODEL"
INCLUDEPRETRARG=""
PRETRMODELARG=""
MINIMALOCCURRENCELIMITARG="--minimal-occurrence-limit 5"
SEPARATEVOCABSARG="--separate-neighbourhood-vocabularies"
if [ ! -z "$USEPRETR" ]
then
    INCLUDEPRETRARG="--include-pretrained-model"
    PRETRMODELARG="--pretrained-model $CONFMODEL"
    MINIMALOCCURRENCELIMITARG=""
    SEPARATEVOCABSARG=""
fi

PRETRWEIGHTSARG="--pretrained-weights bert-base-cased"
if [ ! -z "$PRETRWEIGHTS" ]
then
    PRETRWEIGHTSARG="--pretrained-weights $PRETRWEIGHTS"
fi

BATCHSIZEARG="$BATCHSIZE"
if [ -z "$BATCHSIZE" ]
then
    BATCHSIZEARG="128"
fi

LANGUAGEARG="english"
if [ ! -z "$LANGUAGE" ]
then
    LANGUAGEARG="$LANGUAGE"
fi

SEEDARG="13"
if [ ! -z "$SEED" ]
then
    SEEDARG="$SEED"
fi

PADDINGIDXARG="0"
if [ ! -z "$PADDINGIDX" ]
then
    PADDINGIDXARG="$PADDINGIDX"
fi

srun python3 -u run.py --run-experiments --configuration $CONF --challenge ocr-evaluation --device cuda --seed $SEEDARG --language $LANGUAGEARG --batch-size $BATCHSIZEARG $INCLUDEPRETRARG $PRETRMODELARG --pretrained-model-size 768 --pretrained-max-length 512 $PRETRWEIGHTSARG --padding-idx $PADDINGIDXARG $SEPARATEVOCABSARG $MINIMALOCCURRENCELIMITARG --joint-model --neighbourhood-set-size 1000 --experiment-types neighbourhood-overlap cosine-similarity cosine-distance