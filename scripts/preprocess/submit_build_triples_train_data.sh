#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../../
pwdPath="$(pwd)"

query_path=$pwdPath/data/collections/msmarco-passage/queries.train.tsv
collection_path=$pwdPath/data/collections/msmarco-passage/collection.tsv
triples_ids_path=$pwdPath/data/triples/qidpidtriples.train.full.2.tsv
output_path=$pwdPath/data/preprocess/qtextptexttriples.train.full.2.tsv


python -m src.preprocess.build_triples_train_data \
  --query_path $query_path \
  --collection_path $collection_path \
  --triples_ids_path $triples_ids_path \
  --output_path $output_path
