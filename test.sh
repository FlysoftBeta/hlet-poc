#!/bin/sh
[ ! -f ./val_dataset.jsonl ] && python3 ./gen.py ./val_dataset.jsonl y 1000
[ ! -f preprocessed_val_dataset.pt ] && python3 ./preprocess.py ./val_dataset.jsonl ./preprocessed_val_dataset.pt
python3 ./infer.py ./checkpoint.pt ./preprocessed_val_dataset.pt
