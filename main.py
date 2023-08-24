import torch
import json
import torch.backends.cudnn as cudnn

from config import args
from trainer import Trainer
from logger_config import logger
import time
import datetime








def main():
    start_time = time.time()
    start_datetime = datetime.datetime.now()
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()
    end_datetime = datetime.datetime.now()
    end_time = time.time()
    print(f'start time :{start_datetime} \nend time :{end_datetime} \ntotal time :{end_time - start_time}')


if __name__ == '__main__':
    main()
