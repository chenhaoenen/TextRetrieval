#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../../
pwdPath="$(pwd)"


# filter queries
python $pwdPath/depend/anserini/tools/scripts/msmarco/filter_queries.py \
 --qrels $pwdPath/data/collections/msmarco-passage/qrels.dev.small.tsv \
 --queries $pwdPath/data/collections/msmarco-passage/queries.dev.tsv \
 --output $pwdPath/data/collections/msmarco-passage/queries.dev.small.tsv



# bm25 recall
sh $pwdPath/depend/anserini/target/appassembler/bin/SearchMsmarco \
  -hits 1000 \
  -threads 1 \
  -k1 0.90\
  -b 0.40 \
  -index $pwdPath/data/indexes/msmarco-passage/lucene-index-msmarco \
  -queries $pwdPath/data/collections/msmarco-passage/queries.dev.small.tsv \
  -output $pwdPath/data/runs/run.msmarco-passage.dev.small.tsv


# eval
python $pwdPath/depend/anserini/tools/scripts/msmarco/msmarco_passage_eval.py \
  $pwdPath/data/collections/msmarco-passage/qrels.dev.small.tsv \
  $pwdPath/data/runs/run.msmarco-passage.dev.small.tsv


#####################
#MRR @10: 0.18398616227770961
#QueriesRanked: 6980
#####################



# bm25 recall
sh $pwdPath/depend/anserini/target/appassembler/bin/SearchMsmarco \
  -hits 1000 \
  -threads 1 \
  -index $pwdPath/data/indexes/msmarco-passage/lucene-index-msmarco \
  -queries $pwdPath/data/collections/msmarco-passage/queries.dev.small.tsv \
  -output $pwdPath/data/runs/run.msmarco-passage.dev.small.tsv


# eval
python $pwdPath/depend/anserini/tools/scripts/msmarco/msmarco_passage_eval.py \
  $pwdPath/data/collections/msmarco-passage/qrels.dev.small.tsv \
  $pwdPath/data/runs/run.msmarco-passage.dev.small.tsv

#####################
#MRR @10: 0.18741227770955546
#QueriesRanked: 6980
#####################