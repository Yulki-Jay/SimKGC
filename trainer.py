import glob # 用于查找匹配指定模式的文件名
import json
import torch
import shutil # 用于文件操作和操作系统功能
import wandb

import torch.nn as nn
import torch.utils.data

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW

from doc import Dataset, collate
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger

# 训练过程主要在这里
class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args) # 使用的是AutoTokenizer

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training() # 开始准备多卡训练

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda() # 交叉熵损失函数

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], # 218,964,481
                               lr=args.lr, # args.lr = 5e-5
                               weight_decay=args.weight_decay) # args.weight_decay = 0.0001
        report_num_trainable_parameters(self.model)

        train_dataset = Dataset(path=args.train_path, task=args.task)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps) # 学习率调整
        self.best_metric = None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate, # 定义如何将样本组合成batch,首要目的是控制一个batch的长度一样
            num_workers=args.workers,
            #num_workers=1,
            pin_memory=True, # 将加载的数据存储到固定内存中，以便更快的将数据传递给GPU
            drop_last=True)

        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler() # 使用混合精度进行训练

        for epoch in range(self.args.epochs): # epochs = 50
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        metric_dict = self.eval_epoch(epoch)
        is_best = self.valid_loader and (self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1']) # 是按照hit1来进行选取最好的checkpoint
        if is_best:
            self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        wandb.log({"Eval Acc@1": top1.avg, "Eval Acc@3": top3.avg, "Eval loss": losses.avg})
        return metric_dict

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))

        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data']) # 256

            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict) # batch dict是原始数据
            outputs = ModelOutput(**outputs) # 输出一些数值  应该看一下labels
            logits, labels = outputs.logits, outputs.labels # logits:[bsz,bsz+1] labels:[bsz] 之所以+1，是因为添加了一个自身负样本
            assert logits.size(0) == batch_size
            # head + relation -> tail
            loss = self.criterion(logits, labels)
            # tail -> head + relation
            loss += self.criterion(logits[:, :batch_size].t(), labels)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3)) # 这里没有计算10，应该计算一下hit10
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(outputs.inv_t, 1) # inv_t 是什么，其实我不太了解，我猜测是计算hit10的?
            losses.update(loss.item(), batch_size)
            wandb.log({"Train loss_avg": losses.avg, "Train inv_t_avg":inv_t.avg,"Train top1_avg": top1.avg, "Train top3_avg": top3.avg,"Train loss_val": losses.val, "Train inv_t_val":inv_t.val,"Train top1_val": top1.val, "Train top3_val": top3.val})
            
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp: # 混合精度训练
                self.scaler.scale(loss).backward() # 执行梯度缩放，防止溢出
                self.scaler.unscale_(self.optimizer) # 恢复到原始的梯度
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip) # 梯度剪裁，grad_clip是传递过来的
                self.scaler.step(self.optimizer) # 更新参数，应用梯度缩放
                self.scaler.update() # 更新缩放因子
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step() # 调整学习率

            if i % self.args.print_freq == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1) 
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))
        wandb.log({"Learning rate": self.scheduler.get_last_lr()[0]})

    def _setup_training(self): # 多GPU训练
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda() # 放到多个显卡上面
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps): # 学习率调整
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
