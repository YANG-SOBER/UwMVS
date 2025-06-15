#!/usr/bin/env bash
MVS_TRAINING="/home/gdyang/Downloads/Data/UW_MVS/UwMVS"

python train.py \
--mode="train" \
--dataset=dtu_yao \
--batch_size=4 \
--trainpath=$MVS_TRAINING \
--trainlist lists/dtu/train.txt \
--testlist lists/dtu/val.txt \
--numdepth=192 \
--logdir ./checkpoints/greenish $@
