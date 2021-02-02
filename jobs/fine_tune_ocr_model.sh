#!/bin/bash
#SBATCH --job-name=ft-ocr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
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
if [ ! -z "$USEPRETR" ]
then
    INCLUDEPRETRARG="--include-pretrained-model"
    PRETRMODELARG="--pretrained-model $CONFMODEL"
fi

PRETRWEIGHTSARG="--pretrained-weights bert-base-cased"
if [ ! -z "$PRETRWEIGHTS" ]
then
    PRETRWEIGHTSARG="--pretrained-weights $PRETRWEIGHTS"
fi

LEARNINGRATE="$LR"
if [ -z "$LR" ]
then
    LEARNINGRATE="1e-3"
fi

PATIENCEARG="$PATIENCE"
if [ -z "$PATIENCE" ]
then
    PATIENCEARG="100"
fi

EVALFREQARG="$EVALFREQ"
if [ -z "$EVALFREQ" ]
then
    EVALFREQARG="100"
fi

BATCHSIZEARG="$BATCHSIZE"
if [ -z "$BATCHSIZE" ]
then
    BATCHSIZEARG="2"
fi

LEARNINGRATE="$LR"
if [ -z "$LR" ]
then
    LEARNINGRATE="1e-3"
fi

OUTPUTTYPEARG="$OUTPUTTYPE"
if [ -z "$OUTPUTTYPE" ]
then
    OUTPUTTYPEARG="raw"
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


srun python3 -u run.py --configuration $CONF --challenge ocr-evaluation --epochs 500000 --device cuda --eval-freq $EVALFREQARG --seed $SEEDARG --learning-rate $LEARNINGRATE --skip-validation --metric-types levenshtein-distance jaccard-similarity --language $LANGUAGEARG --batch-size $BATCHSIZEARG --ocr-output-type $OUTPUTTYPEARG --patience $PATIENCEARG $INCLUDEPRETRARG $PRETRMODELARG --pretrained-model-size 768 --pretrained-max-length 512 $PRETRWEIGHTSARG --enable-external-logging --padding-idx $PADDINGIDXARG