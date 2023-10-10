import os
import wandblog
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

import torch
import json
import torch.backends.cudnn as cudnn # 提升训练和推理速度

from config import args # args 传递了所有的参数，已经解析好的了
from trainer import Trainer # 核心都在这里
from logger_config import logger # logger也是一个全局的，可以在任何地方使用，实现日志信息在整个项目中的一致性记录
import time
import datetime


def main():
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    ngpus_per_node = torch.cuda.device_count() # 查看有几个显卡
    cudnn.benchmark = True # 这个需要考虑是否开启，因为在不同的硬件上的性能影响可能不同

    logger.info("Use {} gpus for training".format(ngpus_per_node)) 

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node) # 构造函数
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4))) # 会输出所有的参数
    trainer.train_loop() # 这是核心
    end_datetime = datetime.datetime.now()
    end_time = time.time()
    print(f'start time :{start_datetime} \nend time :{end_datetime} \ntotal time :{end_time - start_time}') # 增加了显示训练时间的功能


if __name__ == '__main__':
    main()
