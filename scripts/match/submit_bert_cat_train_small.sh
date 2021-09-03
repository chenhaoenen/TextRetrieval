#!/bin/bash

currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../../
pwdPath="$(pwd)"

pretrained_model_path=/code/pre_trained_model/model/bert-base-uncased
triple_text_with_label_path=$pwdPath/data/preprocess/querypassagelabeltriples.train.small.tsv

LOG_FILE=$pwdPath/logs/$0.log
rm -f "$LOG_FILE"

run() {
  python -u -m src.match.example.bert_cat_train_small \
    --pretrained_model_path $pretrained_model_path \
    --triple_text_with_label_path $triple_text_with_label_path | tee $LOG_FILE
}

s=`date +'%Y-%m-%d %H:%M:%S'`
run
e=`date +'%Y-%m-%d %H:%M:%S'`
echo '==================================================='
echo "the job start time：$s"
echo "the job  end  time：$e"
echo '==================================================='
