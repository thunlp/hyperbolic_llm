for task in dev test_wsj test_brown train; do
python3 /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/run_rp.py \
--dataset_tag conll2005 \
--train_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2005/role_prediction/train \
--dev_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2005/role_prediction/dev \
--max_tokens 4096 \
--max_epochs 8 \
--lr 7e-6 \
--max_grad_norm 1 \
--warmup_ratio -1 \
--vocab_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/vocab/vocab \
--config_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/bert_large.json \
--save_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large/rp/conll05/7e-6 \
--predict \
--save \
--test_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2005 \
--init_checkpoint /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large_dp=0.1_attdp=0.1_lr=6e-4_step=30000/checkpoints/ckpt_33000.pt \
--dataset_path /apdcephfs/share_47076/weizechen/hilbert/data/download/srl/conll2005/large/${task}.english.psense.conll05.json \
--output_path /apdcephfs/share_47076/weizechen/hilbert/data/download/srl/conll2005/large/${task}.english.psense.plabel.conll05.json \
--tqdm_mininterval 1 \
--alpha 9
done
