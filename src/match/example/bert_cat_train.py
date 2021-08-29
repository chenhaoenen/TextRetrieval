# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/29 11:48 
# Description:  
# --------------------------------------------
import os
import time
import torch
import random
import argparse
from torch import optim
from ..model.bert_cat import BertCat
from src.utils.timer import stats_time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, set_seed, logging

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--query_path', required=True, type=str)
    parser.add_argument('--passage_path', required=True, type=str)
    parser.add_argument('--triple_ids_path', required=True, type=str)
    parser.add_argument('--max_seq_length', default=256, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
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
    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    model = BertCat.from_pretrained(args.pretrained_model_path, config=config).to(device)
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

    # triples
    triples = []
    with open(args.triple_ids_path, 'r') as reader:
        line = reader.readline()
        while line:
            content = line.strip().split('\t')
            assert len(content) == 3
            pid, pos_pid, neg_pid = content
            if pid in querys and pos_pid in passages:
                triples.append((pid, pos_pid, 1))
            if pid in querys and neg_pid in passages:
                triples.append((pid, neg_pid, 0))
            line = reader.readline()

    print(f'load triples num - {len(triples)}')

    # shuffle triples
    random.shuffle(triples)

    class TriplesDataset(Dataset):
        def __init__(self, triples):
            self._data = triples

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            qid, pid, label = self._data[idx]
            return querys[qid], passages[pid], label

    def collate_fn(batch):
        querys, passages, labels = zip(*batch)

        tokenize = tokenizer(querys, passages, max_length=args.max_seq_length, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = tokenize.input_ids
        token_type_ids = tokenize.token_type_ids
        attention_mask = tokenize.attention_mask
        labels = torch.tensor(labels).float()

        return input_ids, token_type_ids, attention_mask, labels

    train_data_loader = DataLoader(dataset=TriplesDataset(triples),
                                   batch_size=args.batch_size,
                                   collate_fn=collate_fn,
                                   num_workers=4)
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
    total_step = args.epochs * len(train_data_loader)
    start = int(time.time())
    step = 0
    for epoch in range(args.epochs):
        #train
        print(f"{'#' * 41} Training {'#' * 41}")
        for i, batch in enumerate(train_data_loader):
            model.train()
            step += 1
            batch = [w.to(device) for w in batch]
            input_ids, token_type_ids, attention_mask, labels = batch
            loss = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

            loss.backward()
            if step % args.log_freq == 0:
                end = int(time.time())
                print(f"epoch:{epoch}, batch:{str(i+1)+'/'+str(len(train_data_loader))}, step:{str(step)+'/'+str(total_step)}, loss:{'{:.6f}'.format(loss)}, eta:{stats_time(start, end, step, total_step)}h, time:{time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())}")
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



