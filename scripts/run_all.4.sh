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
task_name=all
init_checkpoint=${2:-"results/large_dp=0.1_attdp=0.1_lr=6e-4_step=30000/checkpoints/ckpt_33000.pt"}
data_dir=${3:-"/home/chenweize/hilbert/data/download/coref/"}
vocab_file=${4:-"vocab/vocab"}
config_file=${5:-"bert_large.json"}
out_dir=${6:-"results/$task_name"}
num_gpu=${7:-"1"}
batch_size=${8:-"64"}
gradient_accumulation_steps=${9:-"1"}
learning_rate=${10:-"3e-5"}
# hyp_learning_rate="4e-5"
warmup_proportion=${11:-"0.1"}
epochs=${12:-"30"}
precision=${14:-"fp32"}
seed=${15:-"2"}
mode=${16:-"train"}
# margin=${17:-"10"}
max_grad_norm=${18:-"0"}
weight_decay=${19:-"0"}

for learning_rate in 3e-5; do
  for alpha in 5 10 20; do
      for beta in 0.1 0.2 0.4; do
        for weight_decay in 0.01; do
          for max_grad_norm in 1; do

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
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu --master_port 12346"
# fi

CMD="python $mpi_command run_coref.py "
CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"prediction"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"prediction"* ]] ; then
    CMD+="--do_predict "
  fi
  CMD+="--eval_batch_size=128 "
fi

CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "
CMD+="--do_lower_case "
CMD+="--data_dir $data_dir "
CMD+="--bert_model bert-large-uncased "
CMD+="--seed $seed "
CMD+="--alpha_param $alpha "
CMD+="--beta_param $beta "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 128 "
CMD+="--learning_rate $learning_rate "
# CMD+="--hyp_learning_rate $hyp_learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--output_dir $out_dir "
CMD+="--max_grad_norm $max_grad_norm "
CMD+="--weight_decay $weight_decay "
CMD+="$use_fp16"

LOGFILE=$out_dir/logfile

$CMD #|& tee $LOGFILE
# echo -e '\a'
    done
  done
done
done
done
