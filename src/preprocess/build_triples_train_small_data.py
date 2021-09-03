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
    parser.add_argument('--triple_ids_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    return args

def main():

    args = parse_arguments()

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
            query, pos_passage, neg_passage = content
            triples.append((query, pos_passage, 1))
            triples.append((query, neg_passage, 0))
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