import os
import json
import copy
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
from sklearn.metrics import classification_report
from modeling import BertConfig, BertForRolePrediction


def batch_by_tokens(length, max_tokens):
    indexes = []
    i = 0
    while i < len(length):
        for j in range(i, len(length)):
            maxc = max(length[i:j+1])
            maxn = max(maxc, length[min(j+1, len(length)-1)])
            current_batch_tokens = maxc*(j+1-i)
            next_batch_tokens = maxn*(j+2-i)
            if (current_batch_tokens <= max_tokens and next_batch_tokens > max_tokens) or j == len(length)-1:
                indexes.append((i, j))
                i = j+1
                break
    return indexes


class LabelDataset:
    def __init__(self, labels, labels2id, path="", tokenizer=None, max_tokens=512):
        data = json.load(open(path))
        self.data = copy.deepcopy(data)
        self.ids = []
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target = []
        self.labels = labels
        self.labels2id = labels2id
        self.init_data(data, tokenizer, max_tokens)

    def init_data(self, data, tokenizer, max_tokens):
        for s_id, d in enumerate(tqdm(data, desc='preprocessing')):
            sentence = d['sentence']
            arguments = d['arguments']
            predicates = d['predicates']
            sentence1 = [tokenizer.tokenize(w) for w in sentence]
            for i in range(len(predicates)):
                pre = predicates[i]
                args = arguments[i]
                sentence1i = sentence1[:pre]+[['*']] + \
                    sentence1[pre:pre+1]+[['@']]+sentence1[pre+1:]
                sentence2i = ['[CLS]']+sum(sentence1i, [])+['[SEP]']
                input_ids = tokenizer.convert_tokens_to_ids(sentence2i)
                self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                args1 = set()
                for arg in args:
                    s, e, l = arg
                    l = l.split('-')
                    key = l[-1]
                    if key in self.labels:
                        args1.add(key)
                    else:
                        pass
                target = torch.zeros(len(self.labels), dtype=torch.uint8)
                for key in args1:
                    j = self.labels2id[key]
                    target[j] = 1
                self.target.append(target)
                self.ids.append((s_id, i))
        self.do_batch(max_tokens)

    def do_batch(self, max_tokens):
        self.batch_input_ids = []
        self.batch_attention_mask = []
        self.batch_target = []
        length = [len(c) for c in self.input_ids]
        length = np.array(length)
        length[length > max_tokens] = max_tokens
        indexes = batch_by_tokens(length, max_tokens)
        for s, e in tqdm(indexes, desc='batching'):
            input_ids = self.input_ids[s:e+1]
            target = self.target[s:e+1]
            input_ids1 = pad_sequence(input_ids, batch_first=True)
            attention_mask = torch.zeros(input_ids1.shape)
            for i in range(len(input_ids1)):
                attention_mask[i, :len(input_ids[i])] = 1
            target1 = torch.stack(target, dim=0)
            self.batch_input_ids.append(input_ids1)
            self.batch_target.append(target1)
            self.batch_attention_mask.append(attention_mask)

    def predict2json(self, predicts):
        data = self.data
        for d in data:
            d['plabel'] = [[] for _ in d['predicates']]
        for p, _id in zip(predicts, self.ids):
            s_id, p_id = _id
            if p != []:
                data[s_id]['plabel'][p_id].extend(p)
        return data

    def __getitem__(self, i):
        return {'input_ids': self.batch_input_ids[i].long(), 'attention_mask': self.batch_attention_mask[i].float(), 'token_type_ids': torch.zeros_like(self.batch_input_ids[i]).long(), 'target': self.batch_target[i]}

    def __len__(self):
        return len(self.batch_input_ids)


def label_predict(dataset, model, device, amp, labels, labels2id, alpha=-1):
    predict_probs = []
    gold = []
    with torch.no_grad():
        for i in trange(len(dataset), desc='predict'):
            batch = dataset[i]
            input_ids, token_type_ids, attention_mask, target = batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['target']
            input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
            if amp:
                with autocast():
                    predict_prob = model(input_ids, token_type_ids, attention_mask)
            else:
                predict_prob = model(input_ids, token_type_ids, attention_mask)
            gold.append(target)
            predict_probs.append(predict_prob)
    gold1 = torch.cat(gold, dim=0)
    predict_probs1 = torch.cat(predict_probs, dim=0)
    if alpha==-1:
        # evaluation using accuracy, the score is used to select the best checkpoint
        score = classification_report(gold1,(predict_probs1>0.5).int(),output_dict=True,zero_division=0,target_names=labels)['micro avg']
        p,r,f = score['precision'],score['recall'],score['f1-score']
        print('micro avg score: p:{:.4f} r:{:.4f} f:{:.4f}'.format(p,r,f))
        return None   
    # detailed evaluation , used to determine the value of alpha
    predicts = torch.zeros(predict_probs1.shape, dtype=torch.uint8)
    predicts.view(-1)[torch.topk(predict_probs1.float().view(-1),int(len(predicts)*alpha))[1]] = 1
    score = classification_report(gold1,predicts,output_dict=True,zero_division=0,target_names=labels)['micro avg']
    p,r,f = score['precision'],score['recall'],score['f1-score']
    print('micro avg score: p:{:.4f} r:{:.4f} f:{:.4f}'.format(p,r,f))    
    predicts1 = []
    for p in predicts:
        t = []
        for i, pi in enumerate(p):
            if pi == 1:
                t.append(labels[i])
        predicts1.append(t)
    return predicts1


def load_checkpoint(args):
    config = BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForRolePrediction(config)
    save_path = args.save_path + "/best_model.pt"
    print("Loading checkpoint %s" % save_path)
    model.load_state_dict(
        torch.load(save_path, map_location='cpu'),
        strict=True,
    )
    print("Loading completed.")
    model.eval()
    return config, model
