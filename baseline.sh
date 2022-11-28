#!/bin/bash

model=$1     # cvqvae_ema_cls_11_ft_3
name=$2      # random1


######################## Predict SS ########################
echo "Predict SS via ${model} for ${name}.jsonl"
cd src
python train.py -config ../configs/predict_ss.yml               # should change parameters

######################## Get Fasta ########################
echo "Transfer Log file ${model}:sample_top_${name}.txt to ${model}:sample_top_${name}.fasta"
python scripts/calProperty.py -m Log2Fasta \
        -i ../logs/${model}/predict/predict_${name}.txt \
        -o ../logs/${model}/predict/predict_${name}.fasta

######################## Net, Amp and Dist ########################
echo "Eval Sample for ${model}:sample_top_${name}.txt"
python scripts/calProperty.py -m EvalSample \
        -i ../logs/${model}/predict/predict_${name}.txt \
        -o ../logs/${model}/predict/analysis/${name}.jsonl
python scripts/calProperty.py -m Align \
        -i ../raw_data/APD/APD.ss.jsonl,../logs/${model}/predict/analysis/${name}.jsonl \
        -o ../logs/${model}/predict/analysis/${name}.align.jsonl

######################## CAMP ########################
echo "Eval Sample CAMP for ${model}:sample_top_${name}.txt"
# should get response from CAMP
python scripts/calProperty.py -m ParseCAMP \
            -i ../logs/${model}/predict/analysis/${name}.align.camp \
            -o ../logs/${model}/predict/analysis/${name}.align.camp.json
python scripts/calProperty.py -m SelectCand \
            -i ../logs/${model}/predict/analysis/${name}.align.jsonl,../logs/${model}/predict/analysis/${name}.align.camp.json \
            -o ../logs/${model}/predict/analysis/${name}.selected
python scripts/calProperty.py -m SelectCand \
            -i ../logs/${model}/predict/analysis/${name}.align.jsonl \
            -o ../logs/${model}/predict/analysis/${name}.selected
python scripts/calProperty.py -m AnalyseDiversity \
        -i ../raw_data/APD/APD.ss.jsonl,../logs/${model}/predict/analysis/${name}.selected
