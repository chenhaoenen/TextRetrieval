# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/29 12:32 
# Description:  
# --------------------------------------------
import random
import time
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str)
    parser.add_argument('--passage_path', type=str)
    parser.add_argument('--triple_ids_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    return args

def main():

    args = parse_arguments()

    # query
    querys = set()
    with open(args.query_path, 'r') as reader:
        line = reader.readline()
        while line:
            content = line.strip().split('\t')
            assert len(content) == 2
            qid, _ = content
            querys.add(qid)
            line = reader.readline()
    print(f'load query num - {len(querys)}')

    # passage
    passages = set()
    with open(args.passage_path, 'r') as reader:
        line = reader.readline()
        while line:
            content = line.strip().split('\t')
            assert len(content) == 2
            pid, _ = content
            passages.add(pid)
            line = reader.readline()
    print(f'load passage num - {len(passages)}')

    # triples
    writer = open(args.output_path, 'w')

    triples = []
    num = 0
    with open(args.triple_ids_path, 'r') as reader:
        line = reader.readline()
        while line:
            num += 1
            content = line.strip().split('\t')
            assert len(content) == 3
            pid, pos_pid, neg_pid = content
            if pid in querys and pos_pid in passages:
                triples.append((pid, pos_pid, 1))
            if pid in querys and neg_pid in passages:
                triples.append((pid, neg_pid, 0))
            if num % 10000000 == 0:
                print(f"num - {num}  time:{time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())}")
                random.shuffle(triples)
                for q, p, l in triples:
                    writer.write(q + '\t' + p + '\t' + str(l) + '\n')
                writer.flush()
                triples = []
            line = reader.readline()

    random.shuffle(triples)
    for q, p, l in triples:
        writer.write(q + '\t' + p + '\t' + str(l) + '\n')
    writer.flush()
    print(f'load triples num - {num}')


    writer.close()


if __name__ == '__main__':
    main()