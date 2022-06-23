#!/bin/bash

#export FLAGS_fraction_of_gpu_memory_to_use=0.1 
#export MLU_VISIBLE_DEVICES="0,1,2,3"
#export CNNL_GEN_CASE=1
#export CNNL_MIN_VLOG_LEVEL=10
#export NEUWARE_LOGINFO=cnrt
#export CNRT_LOG_LEVEL=3
#export CNRT_MEM_BOUNDRY_CHECK=1
#export MLU_ABORT_AFTER_COREDUMP=1
#export CNNL_GEN_CASE_OP_NAME="embedding_forward;embedding_backward"

# Training with SQuAD v1.1 of dataset
#GLOG_v=10 python -m paddle.distributed.launch --mlus "0" run_squad.py \
python -m paddle.distributed.launch --mlus "3" run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 24 \
    --learning_rate 3e-5 \
    --max_steps 200 \
    --num_train_epochs 1 \
    --logging_steps 100 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./output_bert/1c_fp32 \
    --device mlu \
    --do_train \
    --do_predict \
    --use_profiler False \
    --use_amp False

#-use_amp False 2>&1 | tee log_bert/log_fp32_1c_1

echo "------------ done!!! ----"
