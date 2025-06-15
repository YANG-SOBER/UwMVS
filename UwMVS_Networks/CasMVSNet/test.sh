#!/usr/bin/env bash
TESTPATH="/home/gdyang/Downloads/Data/dtu/Underwater/Greenish"
TESTLIST="./lists/dtu/test.txt"
CKPT_FILE="/home/gdyang/cascade-stereo/CasMVSNet/ckpts/greenish/model_000024.ckpt"
outdir="./outputs_greenish/"

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

CUDA_VISIBLE_DEVICES=0 python test.py \
--dataset=general_eval \
--batch_size=1 \
--outdir=$outdir \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt $CKPT_FILE ${@:2} \
--filter_method gipuma \
--interval_scale 1.06 \
--num_view 5 \
--prob_threshold 0.9 \
--disp_threshold 0.25 \
--num_consistent 4
