#!/bin/sh
# 准备训练数据集
[ ! -f ./dataset.jsonl ] && python3 ./gen.py ./dataset.jsonl n 4000
[ ! -f ./preprocessed_dataset.pt ] && python3 ./preprocess.py ./dataset.jsonl ./preprocessed_dataset.pt

# 生成验证数据集
[ ! -f ./val_dataset.jsonl ] && python3 ./gen.py ./val_dataset.jsonl y 1000
[ ! -f ./preprocessed_val_dataset.pt ] && python3 ./preprocess.py ./val_dataset.jsonl ./preprocessed_val_dataset.pt

# Epochs=80
python3 ./train.py 80 ./checkpoint.pt ./preprocessed_dataset.pt ./preprocessed_val_dataset.pt
