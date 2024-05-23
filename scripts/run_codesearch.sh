#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

echo "Container nvidia build = " $NVIDIA_BUILD_ID
export BERT_PREP_WORKING_DIR="../small-test/data/"
lang=$1
init_checkpoint=${2:-"results/large_dp=0.1_attdp=0.1_lr=6e-4_step=30000/checkpoints/ckpt_33000.pt"}
data_dir=${3:-"/home/chenweize/hilbert/CodeBERT/CodeBERT/data/codesearch/train_valid/$lang"}
vocab_file=${4:-"vocab/vocab"}
config_file=${5:-"bert_large.json"}
out_dir=${6:-"results/codesearch/$lang"}
num_gpu=${7:-"2"}
batch_size=${8:-"32"}
gradient_accumulation_steps=${9:-"1"}
learning_rate=${10:-"1e-5"}
# hyp_learning_rate="4e-5"
warmup_proportion=${11:-"1e-10"}
epochs=${12:-"8"}
precision=${14:-"fp32"}
seed=${15:-"2"}
mode=${16:-"train eval"}
# margin=${17:-"10"}
max_grad_norm=${18:-"1"}
weight_decay=${19:-"0.0"}


# for lr in 5e-6 1e-5 2e-5; do

#   for gn in 0 0.5 1; do
#     for wd in 0 1e-5 1e-2 1e-1; do
#       learning_rate=$lr
#       # hyp_learning_rate=$lr
#       max_grad_norm=$gn
#       weight_decay=$wd

mkdir -p $out_dir

if [ "$mode" = "eval" ] ; then
  num_gpu=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

# if [ "$num_gpu" = "1" ] ; then
#   # export CUDA_VISIBLE_DEVICES=0
#   mpi_command=""
# else
#   unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu --master_port 54332"
# fi

CMD="python $mpi_command run_codesearch.py "
# CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--per_gpu_train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"prediction"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"prediction"* ]] ; then
    CMD+="--do_predict "
  fi
  CMD+="--per_gpu_eval_batch_size=128 "
fi

CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "
CMD+="--do_lower_case "
CMD+="--task_name codesearch "
CMD+="--model_name_or_path hilbert "
CMD+="--data_dir $data_dir "
# CMD+="--bert_model bert-large-uncased "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 200 "
CMD+="--learning_rate $learning_rate "
# CMD+="--hyp_learning_rate $hyp_learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--output_dir $out_dir "
CMD+="--max_grad_norm $max_grad_norm "
CMD+="--weight_decay $weight_decay "
CMD+="--overwrite_output_dir "
CMD+="$use_fp16"

LOGFILE=$out_dir/logfile

# $CMD #|& tee $LOGFILE
# echo -e '\a'
idx=0 #test batch idx

python run_codesearch.py \
--model_name_or_path hilbert \
--task_name codesearch \
--init_checkpoint $init_checkpoint \
--do_predict \
--output_dir ./results/codesearch/$lang \
--data_dir ../CodeBERT/CodeBERT/data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--test_file batch_${idx}.txt \
--vocab_file=$vocab_file \
--config_file=$config_file \
--pred_model_dir ./results/codesearch/$lang/checkpoint-best/ \
--test_result_dir ./results/codesearch/$lang/output/${idx}_batch_result.txt

# lang=php #fine-tuning a language-specific model for each programming language 
# pretrained_model=microsoft/codebert-base  #Roberta: roberta-base

# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 32 \
# --per_gpu_eval_batch_size 32 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models/$lang  \
# --model_name_or_path $pretrained_model
