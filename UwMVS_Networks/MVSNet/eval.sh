#!/usr/bin/env bash
DTU_TESTING="/home/gdyang/Downloads/Data/dtu/Underwater/Greenish"
CKPT_FILE="./checkpoints/greenish/model_000024.ckpt"
out_dir="./outputs/greenish"
CUDA_VISIBLE_DEVICES=1 python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt --outdir=$out_dir --loadckpt $CKPT_FILE $@
