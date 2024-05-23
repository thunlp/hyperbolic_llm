for task in dev test_wsj test_brown train; do
python3 /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/run_rp.py \
--dataset_tag conll2009 \
--train_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009/role_prediction/train \
--dev_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009/role_prediction/dev \
--max_tokens 4096 \
--max_epochs 8 \
--lr 8e-6 \
--max_grad_norm 1 \
--warmup_ratio -1 \
--vocab_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/vocab/vocab \
--config_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/bert_large.json \
--save_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large/rp/conll09/8e-6 \
--predict \
--save \
--test_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009 \
--init_checkpoint /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large_dp=0.1_attdp=0.1_lr=6e-4_step=30000/checkpoints/ckpt_33000.pt \
--dataset_path /apdcephfs/share_47076/weizechen/hilbert/data/download/srl/conll2009/large/${task}.english.psense.conll09.json \
--output_path /apdcephfs/share_47076/weizechen/hilbert/data/download/srl/conll2009/large/${task}.english.psense.plabel.conll09.json \
--tqdm_mininterval 1 \
--alpha 7
done
