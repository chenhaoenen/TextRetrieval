# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/29 12:32 
# Description:  
# --------------------------------------------
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str)
    parser.add_argument('--collection_path', type=str)
    parser.add_argument('--triples_ids_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    return args

def main():

    args = parse_arguments()

    # query
    querys = {}
    with open(args.query_path, 'r') as reader:
        line = reader.readline()
        while line:
            content = line.strip().split('\t')
            assert len(content) == 2
            qid, qtext = content
            querys[qid] = qtext
            line = reader.readline()

    print(f'query num - {len(querys)}')

    # passage
    passages = {}
    with open(args.collection_path, 'r') as reader:
        line = reader.readline()
        while line:
            content = line.strip().split('\t')
            assert len(content) == 2
            pid, ptext = content
            passages[pid] = ptext
            line = reader.readline()

    print(f'passage num - {len(passages)}')


    writer = open(args.output_path, 'w')
    w_num = 0
    with open(args.triples_ids_path, 'r') as reader:
        line = reader.readline()
        while line:
            content = line.strip().split('\t')
            assert len(content) == 3
            qid, p_pid, n_pid = content
            if qid in querys and p_pid in passages and n_pid in passages:
                example = querys[qid] + '\t' + passages[p_pid] + '\t' + passages[n_pid]
                writer.write(example)
                writer.write('\n')
                w_num += 1
            else:
                if qid not in querys:
                    print(f'qid - {qid} not in querys')
                elif p_pid not in passages:
                    print(f'p_pid - {p_pid} not in passages')
                else:
                    print(f'n_pid - {n_pid} not in passages')
            line = reader.readline()
    writer.flush()
    writer.close()
    print(f'output num - {w_num}')


if __name__ == '__main__':
    main()