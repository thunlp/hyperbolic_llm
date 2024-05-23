import time
import os
import random
import argparse
import json

import pickle
from schedulers import PolyWarmUpSchedulerCorrect
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange, tqdm
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

# from model import MyModel
# from evaluate import *
from modeling import BertForPredicateDisambiguation, BertConfig
from utils_srl.pd.dataloader import load_pd_data, reload_pd_data
from utils_srl.pd.evaluation import evaluation_pd
from optim import RiemannianAdam
from tokenization import BertTokenizer
from utils_srl.pd.predict import DisamDataset, disam_predict

from geoopt import ManifoldParameter

import spacy
nlp = spacy.load('en_core_web_sm')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_tag", choices=['conll2005', 'conll2009', 'conll2012'])
    #train_path and dev_path can also be cached data directories.
    parser.add_argument("--train_path")
    parser.add_argument("--dev_path")
    parser.add_argument("--test_path")
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--vocab_file", type=str, default='vocab/vocab')
    parser.add_argument("--config_file", type=str, default="bert_base.json")
    parser.add_argument("--init_checkpoint", default=None, type=str, required=True, help="The checkpoint file from pretraining",
    )

    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--warmup_ratio", type=float, default=-1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1)

    parser.add_argument("--resume", action="store_true",help="used to continue training from the checkpoint")
    parser.add_argument("--checkpoint_path", type=str,help="checkpoint path when resume is true")

    parser.add_argument("--amp", action="store_true",help="whether to enable mixed precision")
    parser.add_argument("--local_rank", type=int, default=-1) ##DDP has been implemented but has not been tested.
    parser.add_argument("--train", action="store_true",help="Whether to train")
    parser.add_argument("--eval", action="store_true",help="Whether to evaluate during training")
    parser.add_argument("--test", action="store_true",help="Whether to test after training")
    parser.add_argument("--predict", action='store_true', help='Whether to predict after training')
    parser.add_argument("--tensorboard", action='store_true',help="whether to use tensorboard to log training information")
    parser.add_argument("--save", action="store_true",help="whether to save the trained model")
    parser.add_argument("--save_path", type=str, default='./checkpoint/')
    parser.add_argument("--tqdm_mininterval", default=1,type=float, help="tqdm minimum update interval")

    parser.add_argument('--frames_path')
    parser.add_argument('--dataset_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    return args


def train(args, train_dataloader, dev_dataloader, resume=False, checkpoint=None):
    config = BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForPredicateDisambiguation(config)
    print("Loading checkpoint %s" % args.init_checkpoint)
    model.load_state_dict(
        torch.load(args.init_checkpoint, map_location='cpu')["model"],
        strict=False,
    )
    print("Loading completed.")
    model.train()
    #prepare training
    if args.amp:
        scaler = GradScaler()
        if resume:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    device = args.local_rank if args.local_rank != -1 else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    if resume:
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
                                                          args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'scale']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay) and not isinstance(p, ManifoldParameter)
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay) or isinstance(p, ManifoldParameter)
            ],
            'weight_decay': 0.0
        },
    ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = RiemannianAdam(optimizer_grouped_parameters, lr=args.lr, stabilize=10, max_grad_norm=args.max_grad_norm)
    if resume:
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state_dict)
    if args.warmup_ratio > 0:
        num_training_steps = len(train_dataloader)*args.max_epochs
        scheduler = PolyWarmUpSchedulerCorrect(
            optimizer, args.warmup_ratio, num_training_steps, degree=1)
        if resume:
            scheduler_state_dict = checkpoint['scheduler_state_dict']
            scheduler.load_state_dict(scheduler_state_dict)
    ltime = time.time()
    start_epoch = 0
    if resume:
        start_epoch = checkpoint['epoch']+1
    #start training
    best_score = 0
    for epoch in range(start_epoch, args.max_epochs):
        if args.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)
        tqdm_train_dataloader = tqdm(train_dataloader, desc="epoch:%d" % epoch, ncols=150, total=len(
            train_dataloader))
        for i, batch in enumerate(tqdm_train_dataloader):
            # torch.cuda.empty_cache()
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, target = batch['input_ids'], batch[
                'token_type_ids'], batch['attention_mask'], batch['target']
            input_ids, token_type_ids, attention_mask, target = input_ids.to(
                device), token_type_ids.to(device), attention_mask.to(device), target.to(device)
            if args.warmup_ratio > 0:
                scheduler.step()
            if args.amp:
                with autocast():
                    loss = model(input_ids, token_type_ids,
                                 attention_mask, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.norm(torch.stack(
                    [torch.norm(p.grad) for p in model.parameters() if p.grad is not None]))
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(input_ids, token_type_ids, attention_mask, target)
                loss.backward()
                grad_norm = torch.norm(torch.stack(
                    [torch.norm(p.grad) for p in model.parameters() if p.grad is not None]))
                # clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            if args.local_rank < 1 and args.tensorboard:
                writer.add_scalar('loss', loss.item(), i + epoch*len(train_dataloader))
                writer.add_scalars("lr_grad", {"lr": lr, "grad_norm": grad_norm}, i+epoch*len(train_dataloader))
                writer.flush()
            # if time.time()-ltime >= args.tqdm_mininterval:
            postfix_str = 'norm:{:.2f},lr:{:.1e},loss:{:.2e}'.format(grad_norm, lr, loss.item())
            tqdm_train_dataloader.set_postfix_str(postfix_str)
            # ltime = time.time()
        if args.local_rank < 1 and args.save:
            if hasattr(model, 'module'):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            checkpoint = {}
            checkpoint['model_state_dict'] = model_state_dict
            checkpoint['optimizer_state_dict'] = optimizer_state_dict
            checkpoint = {"model_state_dict": model_state_dict,
                          "optimizer_state_dict": optimizer_state_dict}
            if args.warmup_ratio > 0:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            if args.amp:
                checkpoint["scaler_state_dict"] = scaler.state_dict()
            save_dir = args.save_path
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                pickle.dump(args, open(save_dir+'/args', 'wb'))
            # save_path = save_dir+"checkpoint_%d.pt" % epoch
            # torch.save(checkpoint, save_path)
            # print("model saved at:", save_path)
        if args.eval and args.local_rank < 1:
            score = evaluation_pd(model,dev_dataloader,args.amp,device)
            if score['accuracy'] > best_score:
                best_score = score['accuracy']
                # save_dir = './checkpoints/%s/disambiguation/' % (args.dataset_tag)
                save_path = args.save_path + "/best_model.pt"
                # torch.save(checkpoint, save_path)
                model_to_save = model
                if hasattr(model, 'module'):
                    model_to_save = model.module
                torch.save(model_to_save.state_dict(), save_path)
                with open(args.save_path + '/args.txt', 'w') as f:
                    f.write(str(args) + '\n')
                print("best model saved at:", save_path)
                with open(os.path.join(args.save_path, 'best_dev.txt'), 'w') as f:
                    f.write(str(best_score))
            if args.tensorboard:
                hp = vars(args)
                hp['epoch']=epoch
                hp['mid']=mid
                writer.add_hparams(hp,score)
                writer.flush()
            model.train()


def test(args, test_dataloader):
    config = BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForPredicateDisambiguation(config)
    save_path = args.save_path + "/best_model.pt"
    print("Loading checkpoint %s" % save_path)
    model.load_state_dict(
        torch.load(save_path, map_location='cpu'),
        strict=True,
    )
    print("Loading completed.")
    model.eval()

    device = args.local_rank if args.local_rank != -1 else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    score = evaluation_pd(model, test_dataloader, args.amp, device)
    return score['accuracy']


if __name__ == "__main__":
    args = args_parser()
    set_seed(args.seed)
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl')
    if args.train:
        if args.train_path.endswith('.json'):
            train_dataloader = load_pd_data(args.vocab_file, args.train_path, args.pretrained_model_name_or_path, args.max_tokens, True, args.local_rank)
            save_dir = os.path.dirname(args.train_path)+'/disambiguation/train/'
            train_dataloader.dataset.save(save_dir)
            print('training data saved at:', save_dir)
        else:
            train_dataloader = reload_pd_data(
                args.train_path, args.max_tokens, True, args.local_rank)
        if not args.eval:
            dev_dataloader = None
        elif args.dev_path.endswith('.json'):
            dev_dataloader = load_pd_data(args.vocab_file, args.dev_path, args.pretrained_model_name_or_path, args.max_tokens, False, -1)
            save_dir = os.path.dirname(args.train_path)+'/disambiguation/dev/'
            dev_dataloader.dataset.save(save_dir)
            print('validation data saved at:', save_dir)
        else:
            dev_dataloader = reload_pd_data(args.dev_path, args.max_tokens, False, -1)
        print(args)
        checkpoint = None
        if args.resume:
            checkpoint = torch.load(args.checkpoint_path,map_location=torch.device('cpu'))
            checkpoint['epoch'] = int(args.checkpoint_path.split('_')[-1].split('.')[0])
        if args.local_rank < 1:
            mid = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
            if args.resume:
                mid = os.path.split(args.checkpoint_path)[-1]
            if args.tensorboard:
                log_dir = "./logs/{}/disambiguation/{}".format(args.dataset_tag, mid)
                writer = SummaryWriter(log_dir)
        train(args, train_dataloader, dev_dataloader, args.resume, checkpoint)

    if args.test and args.local_rank < 1:
        result = {}
        if args.dataset_tag in ['conll2005', 'conll2009']:
            task_list = ['_wsj', '_brown']
        else:
            task_list = ['']
        for task in task_list:
            if args.dataset_tag == 'conll2005':
                tag = 'conll05'
            elif args.dataset_tag == 'conll2009':
                tag = 'conll09'
            else:
                tag = 'conll12'
            test_path = os.path.join(args.test_path, 'test%s.english.%s.json' % (task, tag))
            test_dataloader = load_pd_data(args.vocab_file, test_path, args.pretrained_model_name_or_path, args.max_tokens, False, args.local_rank)
            score = test(args, test_dataloader)
            # print("%s: %f" % (task, score))
            result[task] = score
            writer.add_scalar('%s accuracy' % (task), score)
        for task, acc in result.items():
            print("%s: %f" % (task, acc))
        with open(os.path.join(args.save_path, 'test_result.txt'), 'w') as f:
            for task, res in result.items():
                f.write("%s: %s\n" % (task, str(res)))
    # if args.local_rank < 1 and args.tensorboard:
    #     writer.close()
    if args.predict:
        frames = json.load(open(args.frames_path))
        all_lemmas = sorted(list(set([k.split('.')[0] for k in frames])))
        lemma_dict = {}
        for k, v in frames.items():
            lemma, fid = k.split('.')
            if lemma not in lemma_dict:
                lemma_dict[lemma] = {}
            lemma_dict[lemma][fid] = v['name']
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        # config, model = load_checkpoint(args.checkpoint_path)
        # model.to(device)
        config = BertConfig.from_json_file(args.config_file)
        # Padding for divisibility by 8
        if config.vocab_size % 8 != 0:
            config.vocab_size += 8 - (config.vocab_size % 8)
        model = BertForPredicateDisambiguation(config)
        save_path = args.save_path + "/best_model.pt"
        print("Loading checkpoint %s" % save_path)
        model.load_state_dict(
            torch.load(save_path, map_location='cpu'),
            strict=True,
        )
        print("Loading completed.")
        model.eval()
        model.to(device)
        # tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)
        # tokenizer.add_special_tokens({'additional_special_tokens': ['<p>', '</p>']})
        tokenizer = BertTokenizer(
            args.vocab_file,
            do_lower_case=True,
            max_len=512
        )
        dataset = DisamDataset(lemma_dict, args.dataset_path, tokenizer,args.max_tokens, all_lemmas=all_lemmas)
        predicts, ids = disam_predict(dataset, model, device, args.amp, lemma_dict)
        if args.save:
            data = dataset.predict2json(predicts, ids)
            with open(args.output_path, 'w') as f:
                json.dump(data, f, sort_keys=True, indent=4)
