#!/bin/bash
#SBATCH --job-name=SimKGC-wn18RR          # 作业名称
#SBATCH --output=/home/jiangyunqi/KGC/SimKGC/logs/wn18rr_test_v1.log        # 输出日志的文件名
#SBATCH --mem=50g                 # 任务不限制使用内存
#SBATCH --partition=gpujl          # 队列名称为gpujl


#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)

echo "开始执行"
source /home/jiangyunqi/anaconda3/bin/activate SimKGC      # 激活conda环境
gpustat 

OUTPUT_DIR=./checkpoint/wn18rr/
bash scripts/train_wn.sh


echo "这个工作完成啦"