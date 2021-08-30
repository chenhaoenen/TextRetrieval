#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../../
pwdPath="$(pwd)"

query_path=$pwdPath/data/collections/msmarco-passage/queries.train.tsv
passage_path=$pwdPath/data/collections/msmarco-passage/collection.tsv
triple_ids_path=$pwdPath/data/triples/qidpidtriples.train.full.2.tsv
output_path=$pwdPath/data/preprocess/qidpidlabeltriples.train.full.2.tsv


python -m src.preprocess.build_triples_train_data \
  --query_path $query_path \
  --passage_path $passage_path \
  --triple_ids_path $triple_ids_path \
  --output_path $output_path
