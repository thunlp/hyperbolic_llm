python3 /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/run_rp.py \
--dataset_tag conll2005 \
--train_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2005/role_prediction/train \
--dev_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2005/role_prediction/dev \
--max_tokens 1024 \
--max_epochs 8 \
--lr 2e-5 \
--max_grad_norm 1 \
--warmup_ratio -1 \
--vocab_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/vocab/vocab \
--config_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/bert_base.json \
--save_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/rp/conll05/2e-5 \
--train \
--eval \
--save \
--test \
--test_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2005 \
--init_checkpoint /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/new_base_dp=0.1_attdp=0.1_lr=2e-3_step=30000/checkpoints/ckpt_33000.pt \
--checkpoint_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/srl/rp/conll05/5e-6/best_model.pt  \
--tqdm_mininterval 1 \
--alpha 5 \
# --train_path ../data/download/srl/conll2005/disambiguation/train  \
# --dev_path ../data/download/srl/conll2005/disambiguation/dev  \
