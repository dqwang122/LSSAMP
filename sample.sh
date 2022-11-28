#!/bin/bash

name=$1     # cvqvae_ema_cls_10_ft_2
epoch=$2    # 462
step=$3     # 6700
sample=$4   # 0



######################## Secondary Structure ########################
echo "Analysis Secondary Structure for ${name}:step_${step}.log"
cd tools/prospr
python Helper.py -m analyzeLog -i ../../logs/${name}/train/step_${step}.log
cd ../..


######################## Sample from GT ########################
echo "Convert Data Format for ${name}:epoch_${epoch}_index.pt"
cd src
python preprocess.py -mode index_to_pt \
        -raw_path ../logs/${name}/train/epoch_${epoch}_index.pt \
        -save_path ../logs/${name}/train/epoch_${epoch}_index
python train.py -config ../configs/sample_vqvae.yml                 # should change parameters


######################## Sample from LM ########################
echo "Train and Generate LM for ${name}:epoch_${epoch}_index.pt"
python train.py -config ../configs/pretrain_LM.yml                  # should change parameters * 4
python train.py -config ../configs/generate_LM.yml                  # should change parameters * 4

echo "Convert Sample Sequence for ${name}_LM.step_${sample}.candidate"
python preprocess.py -mode LM_to_pt \
        -raw_path ../logs/${name}_LM \
        -save_path ../logs/${name}_LM  \
        -step ${sample}
python train.py -config ../configs/sample_vqvae_1.yml               # should change parameters


######################## Get Fasta ########################
echo "Transfer Log file for ${name}:sample_top_${sample}.txt"
python scripts/calProperty.py -m Log2Fasta \
        -i ../logs/${name}/sample/sample_top_${sample}.txt \
        -o ../logs/${name}/sample/sample_top_${sample}.fasta


######################## Net, Amp and Dist ########################
echo "Eval Sample for ${name}:sample_top_${sample}.txt"
python scripts/calProperty.py -m EvalSample \
        -i ../logs/${name}/sample/sample_top_${sample}.txt \
        -o ../logs/${name}/sample/sample_top_${sample}.jsonl
python scripts/calProperty.py -m Align \
        -i ../raw_data/APD/APD.ss.jsonl,../logs/${name}/sample/sample_top_${sample}.jsonl \
        -o ../logs/${name}/sample/sample_top_${sample}.align.jsonl


######################## CAMP ########################
echo "Eval Sample CAMP for ${name}:sample_top_${sample}.txt"
# should get response from CAMP
python scripts/calProperty.py -m ParseCAMP \
            -i ../logs/${name}/sample/sample_top_${sample}.align.camp \
            -o ../logs/${name}/sample/sample_top_${sample}.align.camp.json
python scripts/calProperty.py -m SelectCand \
            -i ../logs/${name}/sample/sample_top_${sample}.align.jsonl,../logs/${name}/sample/sample_top_${sample}.align.camp.json \
            -o ../logs/${name}/sample/sample_top_${sample}.selected
python scripts/calProperty.py -m SelectCand \
            -i ../logs/${name}/sample/sample_top_${sample}.align.jsonl \
            -o ../logs/${name}/sample/sample_top_${sample}.selected
python scripts/calProperty.py -m AnalyseDiversity \
        -i ../raw_data/APD/APD.ss.jsonl,../logs/${name}/sample/sample_top_${sample}.selected