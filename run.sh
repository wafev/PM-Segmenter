#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONWARNINGS="ignore"


# eval
#python -m segm.eval.miou \
#/data/sy/segmenter/pretrained/sege_large_mask_ade/checkpoint.pth ade20k --singlescale


# finetune
python -m torch.distributed.launch --master_port=3001 --nproc_per_node=4 --use_env \
prune_finetune.py  --prune --pretrain-dir /data/sy/segmenter/pretrained/seg_base_mask_ade \
--dataset ade20k --log-dir experiments/seg_large_mask_ade_block_08_iter50 \
--backbone vit_base_patch16_384 --decoder mask_transformer --iter-num 1  \
--scheduler cosine

sleep 100
#
python -m torch.distributed.launch --master_port=3001 --nproc_per_node=4 --use_env \
prune_finetune.py  --pretrain-dir experiments/seg_large_mask_ade_block_08_iter50 \
--dataset ade20k --log-dir experiments/seg_large_mask_ade_block_08_iter50 \
--backbone vit_base_patch16_384 --decoder mask_transformer --iter-num 0  \
--scheduler cosine