#!/bin/bash

currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../../
pwdPath="$(pwd)"

pretrained_model_path=/code/pre_trained_model/model/bert-base-uncased
query_path=$pwdPath/data/collections/msmarco-passage/queries.train.tsv
passage_path=$pwdPath/data/collections/msmarco-passage/collection.tsv
triple_ids_with_label_path=$pwdPath/data/preprocess/qidpidlabeltriples.train.full.2.tsv

LOG_FILE=$pwdPath/logs/$0.log
rm -f "$LOG_FILE"

run() {
  python -u -m src.match.example.col_bert_train \
    --pretrained_model_path $pretrained_model_path \
    --query_path $query_path \
    --passage_path $passage_path \
    --triple_ids_with_label_path $triple_ids_with_label_path | tee $LOG_FILE
}

s=`date +'%Y-%m-%d %H:%M:%S'`
run
e=`date +'%Y-%m-%d %H:%M:%S'`
echo '==================================================='
echo "the job start time：$s"
echo "the job  end  time：$e"
echo '==================================================='