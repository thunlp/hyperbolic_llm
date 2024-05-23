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
export BERT_PREP_WORKING_DIR="/apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect"

# if [[ $TASK_CNT == 1 ]]; then
#     MASTER_ADDR=localhost
#     MASTER_PORT=6000
#     NNODES=1
#     NODE_RANK=0
# else
    MASTER_HOST="$ADDRESS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    # NNODES="$TASK_CNT"
    # NODE_RANK="$TASK_INDEX"
    NNODES=16
    NODE_RANK=8
    # echo $BATCH_CUSTOM0_HOSTS
    echo "ADDRESS="$CURRENT_ADDRESS
    echo $ADDRESS
    echo $MASTER_ADDR
    echo $MASTER_PORT
    echo $DLS_TASK_NUMBER
    echo $DLS_TASK_INDEX
# fi

echo "Container nvidia build = " $NVIDIA_BUILD_ID
train_batch_size=${1:-256}
learning_rate=${2:-"1e-3"}
precision=${3:-"fp32"}
num_gpus=${4:-8}
warmup_proportion=${5:-"0.25"}
train_steps=${6:-30000}
save_checkpoint_steps=${7:-500}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-8}
seed=${12:-12439}
job_name=${13:-"bert_adam_pretraining"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"true"}
train_batch_size_phase2=${16:-256}
learning_rate_phase2=${17:-"1e-4"}
warmup_proportion_phase2=${18:-"0.5"}
train_steps_phase2=${19:-3000}
gradient_accumulation_steps_phase2=${20:-64}
DATASET=../data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus # change this for other datasets
DATA_DIR_PHASE1=${21:-$BERT_PREP_WORKING_DIR/${DATASET}/}
BERT_CONFIG=$BERT_PREP_WORKING_DIR/bert_large.json
DATASET2=../data/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus # change this for other datasets
DATA_DIR_PHASE2=${22:-$BERT_PREP_WORKING_DIR/${DATASET2}/}
CODEDIR=${23:-$BERT_PREP_WORKING_DIR}
init_checkpoint=${24:-"None"}
init_loss_scale=${25:-16}
weight_decay_phase1=${26:-0.3}
weight_decay_phase2=${27:-0.3}
poly_degree_phase1=${28:-1}
poly_degree_phase2=${28:-1}
max_grad_norm_phase1=${29:-1}
max_grad_norm_phase2=${30:-1}
RESULTS_DIR=$CODEDIR/results/large_dp=0.1_attdp=0.1_lr=1e-3_step=30000
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints

mkdir -p $CHECKPOINTS_DIR
CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cp $CURR_DIR/${0##*/} $RESULTS_DIR
cp $CODEDIR/modeling.py $RESULTS_DIR
cp $CODEDIR/scripts/run_bert.sh $RESULTS_DIR
cp $CODEDIR/run_pretraining.py $RESULTS_DIR
cp $CODEDIR/bert_large.json $RESULTS_DIR

if [ ! -d "$DATA_DIR_PHASE1" ] ; then
   echo "Warning! $DATA_DIR_PHASE1 directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE1
INPUT_DIR=$DATA_DIR_PHASE1
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --log_freq=10"
# CMD+=" --init_loss_scale=$init_loss_scale"
CMD+=" --weight_decay $weight_decay_phase1"
CMD+=" --max_grad_norm $max_grad_norm_phase1"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nnodes $NNODES --node_rank $NODE_RANK $CMD"
# CMD="python3 $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.${NODE_RANK}.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) 2>&1 |& tee $LOGFILE
fi

set +x

echo "finished pretraining"
echo -e '\a'

#Start Phase2

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_phase2"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

echo $DATA_DIR_PHASE2
INPUT_DIR=$DATA_DIR_PHASE2
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE2"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size_phase2"
CMD+=" --max_seq_length=512"
CMD+=" --poly_degree $poly_degree_phase2"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --log_freq=10"
CMD+=" --weight_decay $weight_decay_phase2"
CMD+=" --max_grad_norm $max_grad_norm_phase2"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" --do_train --phase2 --resume_from_checkpoint --phase1_end_step=$train_steps"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nnodes $NNODES --node_rank $NODE_RANK $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.${NODE_RANK}.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) 2>&1 |& tee $LOGFILE
fi

set +x

echo "finished phase2"
echo -e '\a'
