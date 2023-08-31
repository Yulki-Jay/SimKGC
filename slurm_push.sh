#!/bin/bash
#SBATCH --job-name=SimKGC-wn18RR          # 作业名称
#SBATCH --output=/home/jiangyunqi/KGC/SimKGC/logs/wn18rr_evaluate_v1.log        # 输出日志的文件名
#SBATCH --mem=0                   # 任务不限制使用内存
#SBATCH --partition=gpujl          # 队列名称为gpujl


#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --mem=0

echo "开始评估性能"
source /home/jiangyunqi/anaconda3/bin/activate SimKGC      # 激活conda环境
gpustat 

bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR

echo "这个工作完成啦"