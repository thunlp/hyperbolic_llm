python3 /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/run_al.py \
--dataset_tag conll2009 \
--train_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009/arg_labeling/1_query_type2.1/train \
--dev_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009/arg_labeling/1_query_type2.1/dev \
--max_tokens 1024 \
--max_epochs 20 \
--lr 9e-6 \
--max_grad_norm 1 \
--warmup_ratio 0.01 \
--arg_query_type 2 \
--argm_query_type 1 \
--gold_level 1 \
--vocab_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/vocab/vocab \
--config_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/bert_base.json \
--save_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/al/conll09/9e-6 \
--train \
--eval \
--save \
--test \
--test_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009 \
--tensorboard \
--init_checkpoint /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/new_base_dp=0.1_attdp=0.1_lr=2e-3_step=30000/checkpoints/ckpt_33000.pt \
--tqdm_mininterval 1
