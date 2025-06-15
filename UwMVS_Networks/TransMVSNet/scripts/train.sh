#!/usr/bin/env bash
# run this script in the root path of TransMVSNet
MVS_TRAINING="/home/gdyang/Downloads/Data/UW_MVS/UwMVS" # path to dataset mvs_training
LOG_DIR="./ckpts/greenish" # path to checkpoints
if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi

NGPUS=1
BATCH_SIZE=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
	--logdir=$LOG_DIR \
	--dataset=dtu_yao \
	--batch_size=$BATCH_SIZE \
	--epochs=16 \
	--trainpath=$MVS_TRAINING \
	--trainlist=lists/dtu/train.txt \
	--testlist=lists/dtu/val.txt \
	--numdepth=192 \
	--ndepths="48,32,8" \
	--nviews=5 \
	--wd=0.0001 \
	--depth_inter_r="4.0,1.0,0.5" \
	--lrepochs="6,8,12:2" \
	--dlossw="1.0,1.0,1.0" | tee -a $LOG_DIR/log.txt
