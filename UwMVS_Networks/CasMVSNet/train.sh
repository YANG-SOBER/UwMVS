#!/usr/bin/env bash
MVS_TRAINING="/home/gdyang/Downloads/Data/UW_MVS/dtu"

LOG_DIR="./ckpts/greenish/"

CKPT_FILE="./ckpts/greenish/model_000024.ckpt"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

CUDA_VISIBLE_DEVICES=1 python train.py \
--mode="test" \
--loadckpt=$CKPT_FILE \
--logdir $LOG_DIR \
--dataset=dtu_yao \
--batch_size=1 \
--trainpath=$MVS_TRAINING \
--trainlist lists/dtu/train.txt \
--testlist lists/dtu/val.txt \
--numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt
