# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/29 11:48 
# Description:  
# --------------------------------------------
import os
import time
import torch
import argparse
import subprocess
from torch import optim
from ..model.bert_dot import BertDot
from src.utils.timer import stats_time
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, set_seed, logging

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--query_path', required=True, type=str)
    parser.add_argument('--passage_path', required=True, type=str)
    parser.add_argument('--triple_ids_with_label_path', required=True, type=str)
    parser.add_argument('--max_seq_length', default=256, type=int)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--learning_rate', default=1.5e-3, type=float)
    parser.add_argument('--log_freq', default=100, type=int)

    args = parser.parse_args()

    return args

def setup_training(args):
    set_seed(args.seed)
    logging.set_verbosity_error()
    assert torch.cuda.is_available()
    device = torch.device('cuda:1')
    args.device = device
    assert os.path.isdir(args.pretrained_model_path), f"pre-training model path:{args.pre_trained_model_path} is not exists"

    os.environ['TOKENIZERS_PARALLELISM'] = "true" #huggingface tokenizer ignore warning

    return args, device

def prepare_model_and_optimizer(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    model = BertDot(args.pretrained_model_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, optimizer, tokenizer

def triples_data_loader(args, tokenizer):
    '''msmarco_passage triples train data set'''

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
    print(f'load query num - {len(querys)}')

    # passage
    passages = {}
    with open(args.passage_path, 'r') as reader:
        line = reader.readline()
        while line:
            content = line.strip().split('\t')
            assert len(content) == 2
            pid, ptext = content
            passages[pid] = ptext
            line = reader.readline()
    print(f'load passage num - {len(passages)}')

    #tiples
    args.train_data_num = int(subprocess.check_output("wc -l " + args.triple_ids_with_label_path, shell=True).split()[0])
    print(f'load train data num - {args.train_data_num}')

    class TriplesDataset(IterableDataset):
        def __init__(self, filename):
            self._file_name = filename

        def line_mapper(self, line):
            qid, pid, label = line.strip().split('\t')
            return querys[qid], passages[pid], float(label)

        def __iter__(self):
            line = open(self._file_name)
            return map(self.line_mapper, line)


    def collate_fn(batch):
        querys, passages, labels = zip(*batch)
        querys_tokenize = tokenizer(list(querys), max_length=30, truncation=True, padding='max_length', return_tensors='pt')
        passages_tokenize = tokenizer(list(passages), max_length=200, truncation=True, padding='max_length', return_tensors='pt')
        labels = torch.tensor(labels)

        return querys_tokenize, passages_tokenize, labels

    train_data_loader = DataLoader(dataset=TriplesDataset(args.triple_ids_with_label_path),
                                   batch_size=args.batch_size,
                                   collate_fn=collate_fn)
    return train_data_loader


# def eval_corrcoef(senta_embeds, sentb_embeds, labels):
#     senta_vecs = l2_normalize(senta_embeds)
#     sentb_vecs = l2_normalize(sentb_embeds)
#
#     sims = (senta_vecs * sentb_vecs).sum(axis=1)
#     corrcoef = compute_corrcoef(labels, sims)
#
#     return corrcoef

def trainer():
    args = parse_arguments()
    args, device = setup_training(args)

    model, optimizer, tokenizer = prepare_model_and_optimizer(args, device)
    train_data_loader = triples_data_loader(args, tokenizer)

    # train_data_loader, (senta_eval_data_loader, sentb_eval_data_loader, labels) = sentevalCNDataLoader(args, tokenizer)
#
    print(f"{'#' * 43} Args {'#' * 43}")
    for k in list(vars(args).keys()):
        print('{0}: {1}'.format(k, vars(args)[k]))
#
    every_epoch_step = args.train_data_num // args.batch_size
    total_step = args.epochs * every_epoch_step
    start = int(time.time())
    step = 0
    for epoch in range(args.epochs):
        #train
        print(f"{'#' * 41} Training {'#' * 41}")
        for i, batch in enumerate(train_data_loader):
            model.train()
            step += 1
            batch = [w.to(device) for w in batch]
            querys, passages, labels = batch
            loss = model(querys=querys, passages=passages, labels=labels)

            loss.backward()
            if step % args.log_freq == 0:
                end = int(time.time())
                print(f"epoch:{epoch}, batch:{str(i+1)+'/'+str(every_epoch_step)}, step:{str(step)+'/'+str(total_step)}, loss:{'{:.6f}'.format(loss)}, eta:{stats_time(start, end, step, total_step)}h, time:{time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())}")
            optimizer.step()
            optimizer.zero_grad()
#
#         print(f"{'#' * 40} Evaluating {'#' * 40}")
#         model.eval()
#
#         #senta and sentb embedding
#         senta_embeds = []
#         sentb_embeds = []
#         with torch.no_grad():
#             for batch in senta_eval_data_loader:
#                 batch = [w.to(device) for w in batch]
#                 input_ids, token_type_ids, attention_mask = batch
#                 embed = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#                 senta_embeds.append(embed.cpu())
#             for batch in sentb_eval_data_loader:
#                 batch = [w.to(device) for w in batch]
#                 input_ids, token_type_ids, attention_mask = batch
#                 embed = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#                 sentb_embeds.append(embed.cpu())
#         senta_embeds = torch.cat(senta_embeds, dim=0).numpy()
#         sentb_embeds = torch.cat(sentb_embeds, dim=0).numpy()
#
#         print(f'task_name:{args.task_name}, corrcoef:{eval_corrcoef(senta_embeds, sentb_embeds, labels)}')


if __name__ == '__main__':
    trainer()



