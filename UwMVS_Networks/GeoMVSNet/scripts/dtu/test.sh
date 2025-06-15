#!/usr/bin/env bash

BESTEPOCH="24"

LOG_DIR="./checkpoints/dtu/greenish"
DTU_CKPT_FILE=$LOG_DIR"/model_"$BESTEPOCH".ckpt"
DTU_OUT_DIR="./outputs/dtu/greenish"
DTU_TEST_ROOT="/home/gdyang/Downloads/Data/dtu/Underwater/Greenish"

CUDA_VISIBLE_DEVICES=0 python3 test.py ${@} \
    --which_dataset="dtu" --loadckpt=$DTU_CKPT_FILE --batch_size=1 \
    --outdir=$DTU_OUT_DIR --logdir=$LOG_DIR --nolog \
    --testpath=$DTU_TEST_ROOT --testlist="datasets/lists/dtu/test.txt" \
    \
    --data_scale="raw" --n_views="5"
