from tokenization import BertTokenizer
from fewnerd.util.data_loader import get_loader
from fewnerd.util.framework import FewShotNERFramework
from fewnerd.util.word_encoder import BERTWordEncoder
from fewnerd.model.hyp_proto import HypProto
from fewnerd.model.hyp_tagger import HypTagger
import dllogger
import logging
from utils import (is_main_process, mkdir_by_main_process, format_step,
                   get_world_size)
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import torch
import random

logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../small-test/data/few-nerd',
            help='training mode, must be in [inter, intra, supervised]')
    parser.add_argument('--mode', default='inter',
            help='training mode, must be in [inter, intra, supervised]')
    parser.add_argument('--trainN', default=2, type=int,
            help='N in train')
    parser.add_argument('--N', default=2, type=int,
            help='N way')
    parser.add_argument('--K', default=2, type=int,
            help='K shot')
    parser.add_argument('--Q', default=3, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=600, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=100, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=500, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=20, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='hyp_proto',
            help='model name, must be basic-bert, proto, hyp_proto, nnshot, or structshot')
    parser.add_argument('--max_length', default=100, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-4, type=float,
           help='learning rate')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    # parser.add_argument('--ckpt_name', type=str, default='',
    #        help='checkpoint name.')
    parser.add_argument('--vocab_file', type=str, default='',
           help='vocab file name.')
    parser.add_argument('--init_checkpoint', type=str, default='',
           help='checkpoint name.')
    parser.add_argument('--config_file', type=str, default='',
           help='config file name.')
    parser.add_argument('--output_dir', type=str, default='results/fewnerd/',
           help='config file name.')
    parser.add_argument('--seed', type=int, default=0,
           help='random seed')
    parser.add_argument('--ignore_index', type=int, default=-1,
           help='label index to ignore when calculating loss and metrics')


    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')

    # only for structshot
    parser.add_argument('--tau', default=0.05, type=float,
           help='StructShot parameter to re-normalizes the transition probabilities')

    # experiment
    parser.add_argument('--use_sgd_for_bert', action='store_true',
           help='use SGD instead of AdamW for BERT.')

    opt = parser.parse_args()

    if is_main_process():
        if (os.path.exists(opt.output_dir) and os.listdir(opt.output_dir)):
            logger.warning("Output directory ({}) already exists and is not "
                           "empty.".format(opt.output_dir))
    mkdir_by_main_process(opt.output_dir)

    if is_main_process():
        dllogger.init(backends=[
            dllogger.JSONStreamBackend(
                verbosity=dllogger.Verbosity.VERBOSE,
                filename=os.path.join(opt.output_dir, 'dllogger.json'),
            ),
            dllogger.StdOutBackend(
                verbosity=dllogger.Verbosity.VERBOSE,
                step_format=format_step,
            ),
        ])
    else:
        dllogger.init(backends=[])

    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    max_length = opt.max_length

    print("{}-way-{}-shot Few-Shot NER".format(N, K))
    print("model: {}".format(model_name))
    print("max_length: {}".format(max_length))
    print('mode: {}'.format(opt.mode))

    set_seed(opt.seed)
    print('loading model and tokenizer...')
    # pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
    word_encoder = BERTWordEncoder(opt)
    tokenizer = tokenizer = BertTokenizer(
        opt.vocab_file,
        do_lower_case=True,
        max_len=512,
    )

    print('loading data...')
    opt.train = f'{opt.data_dir}/{opt.mode}/train.txt'
    opt.test = f'{opt.data_dir}/{opt.mode}/test.txt'
    opt.dev = f'{opt.data_dir}/{opt.mode}/dev.txt'
    if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
        os.system(f'bash data/download.sh {opt.mode}')

    train_data_loader = get_loader(opt.train, tokenizer,
            N=trainN, K=K, Q=Q, batch_size=batch_size, max_length=max_length, ignore_index=opt.ignore_index)
    val_data_loader = get_loader(opt.dev, tokenizer,
            N=N, K=K, Q=Q, batch_size=batch_size, max_length=max_length, ignore_index=opt.ignore_index)
    test_data_loader = get_loader(opt.test, tokenizer,
            N=N, K=K, Q=Q, batch_size=batch_size, max_length=max_length, ignore_index=opt.ignore_index)

        
    prefix = '-'.join([model_name, opt.mode, str(N), str(K), 'seed'+str(opt.seed)])
    if opt.dot:
        prefix += '-dot'
    # if len(opt.ckpt_name) > 0:
    #     prefix += '-' + opt.ckpt_name
    
    if model_name == 'hyp_proto':
        print('use hyp_proto')
        model = HypProto(word_encoder, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, mode=opt.mode)
    elif model_name == 'hyp_tag':
        print('use hyp_tag')
        model = HypTagger(word_encoder, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, mode=opt.mode)
    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt
    print('model-save-path:', ckpt)

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if opt.lr == -1:
            opt.lr = 2e-5

        framework.train(model, prefix,
                load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                val_step=opt.val_step, fp16=opt.fp16,
                train_iter=opt.train_iter, warmup_step=int(opt.train_iter * 0.1), val_iter=opt.val_iter, learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    # test
    precision, recall, f1, fp, fn, within, outer = framework.eval(model, opt.test_iter, ckpt=ckpt)
    print("RESULT: precision: %.4f, recall: %.4f, f1:%.4f" % (precision, recall, f1))
    print('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within:%.4f, outer: %.4f'%(fp, fn, within, outer))

if __name__ == "__main__":
    main()
