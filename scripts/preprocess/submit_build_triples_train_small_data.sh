#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../../
pwdPath="$(pwd)"


triple_ids_path=$pwdPath/data/triples/triples.train.small.tsv
output_path=$pwdPath/data/preprocess/querypassagelabeltriples.train.small.tsv


python -m src.preprocess.build_triples_train_small_data \
  --triple_ids_path $triple_ids_path \
  --output_path $output_path
