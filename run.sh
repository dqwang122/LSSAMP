#!/usr/bin/env bash

HDFSPREF=HDFS

CONFIG=$1
NAME=$2

echo "Build environment..."
mkdir models logs configs
local_model=peptide/models/$NAME
local_logs=peptide/logs/$NAME
mkdir $local_model $local_logs

echo "Download resource..."
hdfs dfs -get ${HDFSPREF}/Datasets/Drugs/tokenized/* data
hdfs dfs -get ${HDFSPREF}/Workshop/Drug/peptide/configs/* configs

echo "Watch output dir..."
hdfs_model_path=${HDFSPREF}/Workshop/Drug/peptide/models/$NAME
hdfs dfs -mkdir $hdfs_model_path
(inotifywait -m ${local_model} -e close_write |
    while read path action file; do
        if [[ "$file" =~ .*st.pt$ ]]; then
            echo "Checkpoint detected: $file" >&2
            hadoop fs -put -f ${local_model}/$file ${hdfs_model_path}
        fi
    done) &


echo "Run training..."
cd src
python train.py -config ../configs/$CONFIG


echo "Watch output dir..."
hdfs_log_path=${HDFSPREF}/Workshop/Drug/peptide/logs/$NAME
hdfs dfs -mkdir $hdfs_log_path
hdfs dfs -put $local_logs/* $hdfs_log_path
