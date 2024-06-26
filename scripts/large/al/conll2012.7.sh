python3 /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/run_al.py \
--dataset_tag conll2012 \
--train_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2012/large/arg_labeling/1_query_type2.1/train \
--dev_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2012/large/arg_labeling/1_query_type2.1/dev \
--max_tokens 2048 \
--max_epochs 20 \
--lr 2e-5 \
--max_grad_norm 1 \
--warmup_ratio 0.01 \
--arg_query_type 2 \
--argm_query_type 1 \
--gold_level 1 \
--vocab_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/vocab/vocab \
--config_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/bert_large.json \
--save_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large/al/conll12/2e-5 \
--train \
--eval \
--save \
--test \
--test_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2012 \
--tensorboard \
--init_checkpoint /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large_dp=0.1_attdp=0.1_lr=6e-4_step=30000/checkpoints/ckpt_33000.pt \
--tqdm_mininterval 1
