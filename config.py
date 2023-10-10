import os
import random
import torch
import argparse
import warnings

import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='SimKGC arguments')
parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--task', default='wn18rr', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--train-path', default='', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--model-dir', default='', type=str, metavar='N',
                    help='path to model dir')
parser.add_argument('--warmup', default=400, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--max-to-keep', default=5, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--pooling', default='cls', type=str, metavar='N',
                    help='bert pooling')
parser.add_argument('--dropout', default=0.1, type=float, metavar='N',
                    help='dropout on final linear layer')
parser.add_argument('--use-amp', action='store_true', # 自动混合精度计算
                    help='Use amp if available')
parser.add_argument('--t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--use-link-graph', action='store_true', # 这个应该仔细看一下
                    help='use neighbors from link graph as context')
parser.add_argument('--eval-every-n-step', default=10000, type=int,
                    help='evaluate every n steps')
parser.add_argument('--pre-batch', default=0, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                    help='the weight for logits from pre-batch negatives')
parser.add_argument('--additive-margin', default=0.0, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--finetune-t', action='store_true',
                    help='make temperature as a trainable parameter or not')
parser.add_argument('--max-num-tokens', default=50, type=int,
                    help='maximum number of tokens')
parser.add_argument('--use-self-negative', action='store_true',
                    help='use head entity as negative')

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# only used for evaluation,这一部分还不确定作用是什么
parser.add_argument('--is-test', action='store_true',
                    help='is in test mode or not')
parser.add_argument('--rerank-n-hop', default=2, type=int,
                    help='use n-hops node for re-ranking entities, only used during evaluation')
parser.add_argument('--neighbor-weight', default=0.0, type=float,
                    help='weight for re-ranking entities')
parser.add_argument('--eval-model-path', default='', type=str, metavar='N',
                    help='path to model, only used for evaluation')

args = parser.parse_args() # 解析了所有的参数

from datetime import datetime

def modify_args(args): # 新添加的函数，方便进行调试，超参数只要修改这就可以了
    dir = os.getcwd()
    args.model_dir = os.path.join(dir, 'checkpoint', '{}_{}'.format(
    'wn18rr','avc'))
    args.pretrained_model = 'bert-base-uncased'
    args.pooling='mean'
    args.lr=5e-5
    args.use_link_graph=True
    args.train_path='//home//jiangyunqi//KGC//SimKGC//data//WN18RR//train.txt.json'
    args.valid_path='//home//jiangyunqi/KGC//SimKGC//data//WN18RR//valid.txt.json'
    args.task='wn18rr'
    args.batch_size=256 # 1024 可以在4张卡，一张卡会爆显存
    args.print_freq=20
    args.additive_margin=0.02
    args.use_amp=True
    args.use_self_negative=True
    args.pre_batch=0
    args.finetune_t=True
    args.epochs=50
    args.workers=4
    args.max_to_keep=3
    print(args.model_dir) 
    return args
# args = modify_args(args)


assert not args.train_path or os.path.exists(args.train_path)
assert args.pooling in ['cls', 'mean', 'max']
assert args.task.lower() in ['wn18rr', 'fb15k237', 'wiki5m_ind', 'wiki5m_trans']
assert args.lr_scheduler in ['linear', 'cosine']

if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True) # 目录如果已经存在的话，不要跑出错误，而是忽略
else:
    assert os.path.exists(args.eval_model_path), 'One of args.model_dir and args.eval_model_path should be valid path'
    args.model_dir = os.path.dirname(args.eval_model_path)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True # 确保了在相同条件下，每次运行模型时都会获得相同的结果，但是会导致性能上的损失，尽量在训练的过程中不启动他

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn('GPU is not available, set use_amp=False and print_freq=1')



def mian():
    print(args.task)
    
if __name__ == '__main__':
    mian()