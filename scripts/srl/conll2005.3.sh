python3 /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/run_srl.py \
--dataset_tag conll2005 \
--train_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2005/disambiguation/train \
--dev_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2005/disambiguation/dev \
--max_tokens 2048 \
--max_epochs 6 \
--lr 8e-6 \
--max_grad_norm 1 \
--warmup_ratio -1 \
--vocab_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/vocab/vocab \
--config_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/bert_base.json \
--save_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/srl/conll05/8e-6 \
--train \
--eval \
--save \
--test \
--test_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2005 \
--tensorboard \
--init_checkpoint /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/new_base_dp=0.1_attdp=0.1_lr=2e-3_step=30000/checkpoints/ckpt_33000.pt \
--tqdm_mininterval 1
# --train_path ../data/download/srl/conll2009/disambiguation/train  \
# --dev_path ../data/download/srl/conll2009/disambiguation/dev  \
