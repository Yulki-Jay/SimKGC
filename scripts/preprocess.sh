#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"
if [[ $# -ge 1 ]]; then
    TASK=$1
    shift
fi

python3 -u preprocess.py \
--task "${TASK}" \
--train-path "./data/${TASK}/train.txt" \
--valid-path "./data/${TASK}/valid.txt" \
--test-path "./data/${TASK}/test.txt"

# 这里面只有4个超参数 task用来决定是使用哪个数据集