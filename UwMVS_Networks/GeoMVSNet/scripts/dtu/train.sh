#!/usr/bin/env bash
# source scripts/data_path.sh


THISNAME="greenish"

DTU_TRAIN_ROOT="/home/gdyang/Downloads/Data/UW_MVS/UwMVS"

LOG_DIR="./ckpts/"$THISNAME

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi


CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --nproc_per_node=1 train.py ${@} \
    --which_dataset="dtu" --epochs=25 --logdir=$LOG_DIR \
    --trainpath=$DTU_TRAIN_ROOT --testpath=$DTU_TRAIN_ROOT \
    --trainlist="datasets/lists/dtu/train.txt" --testlist="datasets/lists/dtu/val.txt" \
    \
    --data_scale="mid" --n_views="5" --batch_size=2 --lr=0.002 --robust_train \
    --lrepochs="1,3,5,7,9,11,13,15:1.5"
