#!/usr/bin/env bash
TESTPATH="/home/gdyang/Downloads/Data/dtu/Underwater/Greenish" 						# path to dataset dtu_test
TESTLIST="./lists/dtu/test.txt"
CKPT_FILE="./ckpts/greenish/model_000024.ckpt"			   # path to checkpoint file, you need to use the model_dtu.ckpt for testing
FUSIBLE_PATH="./fusibile/fusibile" 								 	# path to fusible of gipuma
OUTDIR="./outputs/greenish" 						  # path to output

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi


CUDA_VISIBLE_DEVICES=1 python test.py \
--dataset=general_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--numdepth=192 \
--max_h=864 \
--max_w=1152 \
--ndepths="48,32,8" \
--depth_inter_r="4.0,1.0,0.5" \
--interval_scale=1.06 \
--filter_method="gipuma" \
--fusibile_exe_path=$FUSIBLE_PATH
