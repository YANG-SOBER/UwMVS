#!/usr/bin/env bash
datapath="/home/gdyang/Downloads/Data/UW_MVS/UwMVS"

log_dir="./ckpts/greenish"

if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi

python -m torch.distributed.launch --nproc_per_node=2 main.py \
        --val \
        --sync_bn \
        --ndepths 48 32 8 \
        --interval_ratio 4 2 1 \
        --img_size 512 640 \
        --num_view 5 \
        --dlossw 0.5 1.0 2.0 \
        --log_dir $log_dir \
        --datapath $datapath \
        --dataset_name "dtu_yao" \
        --epochs 25 \
        --batch_size 2 \
        --lr 0.001 \
        --warmup 0.2 \
        --scheduler "steplr" \
        --milestones 10 12 14 \
        --lr_decay 0.5 \
        --trainlist "datasets/lists/dtu/train.txt" \
        --testlist "datasets/lists/dtu/val.txt" \
        --fea_mode "fpn" \
        --agg_mode "adaptive" \
        --depth_mode "unification" \
        --numdepth 192 \
        --interval_scale 1.06 ${@:1} | tee -a $log_dir/log.txt
