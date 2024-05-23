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
export BERT_PREP_WORKING_DIR="../data/download/"

init_checkpoint="results/large_dp=0.1_attdp=0.1_lr=6e-4_step=30000/checkpoints/ckpt_33000.pt"
data_dir="$BERT_PREP_WORKING_DIR/glue/$1/"
vocab_file="vocab/vocab"
config_file="bert_large.json"
out_dir="results/large/$1"
task_name="$1"
num_gpu="1"
batch_size="32"
gradient_accumulation_steps="1"
learning_rate="2e-5"
warmup_proportion="0.1"
epochs="3"
max_steps="-1.0"
precision="fp32"
seed="2"
mode="train eval"
margin="10"
max_grad_norm="1"
weight_decay="0.01"
eval_step=50

if [ "$task_name" = "rte" ] ; then
  eval_step=8
elif [ "$task_name" = "mnli" ]; then
  eval_step=1227
elif [ "$task_name" = "qqp" ]; then
  eval_step=1137
elif [ "$task_name" = "qnli" ]; then
  eval_step=50
elif [ "$task_name" = "sst-2" ]; then
  eval_step=210
elif [ "$task_name" = "cola" ]; then
  eval_step=26
elif [ "$task_name" = "mrpc" ]; then
  eval_step=11
elif [ "$task_name" = "sts-b" ]; then
  eval_step=18
fi

if [ "$mode" = "eval" ] ; then
  num_gpu=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

for learning_rate in 1e-5 2e-5 3e-5 4e-5 5e-5; do
  for seed in 0 1 2 3 4; do
  out_dir=results/$1/lr${learning_rate}/seed${seed}
  mkdir -p $out_dir
if [ "$num_gpu" = "1" ] ; then
  # export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

CMD="python $mpi_command run_glue.py "
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
  CMD+="--eval_batch_size=$batch_size "
fi

CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "
CMD+="--do_lower_case "
CMD+="--data_dir $data_dir "
CMD+="--bert_model bert-large-uncased "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 128 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--output_dir $out_dir "
CMD+="--margin $margin "
CMD+="--max_grad_norm $max_grad_norm "
CMD+="--weight_decay $weight_decay "
CMD+="--eval_step $eval_step "
CMD+="$use_fp16"

LOGFILE=$out_dir/log

$CMD |& tee $LOGFILE
done
done
echo -e '\a'
