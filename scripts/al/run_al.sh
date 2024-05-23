python3 run_al.py \
--dataset_tag conll2009 \
--train_path ../data/download/srl/conll2009/arg_labeling/1_query_type2.1/train/ \
--dev_path ../data/download/srl/conll2009/arg_labeling/1_query_type2.1/dev/ \
--max_tokens 2048 \
--max_epochs 20 \
--lr 1e-5 \
--max_grad_norm 1 \
--warmup_ratio 0.01 \
--arg_query_type 2 \
--argm_query_type 1 \
--gold_level 1 \
--tensorboard \
--train \
--eval \
--save \
--save_path results/srl/al/conll09/1e-5 \
--init_checkpoint ./results/new_base_dp=0.1_attdp=0.1_lr=2e-3_step=30000/checkpoints/ckpt_33000.pt \
--tqdm_mininterval 1 
