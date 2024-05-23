for task in dev test_wsj test_brown train; do
python3 /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/run_srl.py \
--dataset_tag conll2009 \
--train_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009/disambiguation/train \
--dev_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009/disambiguation/dev \
--max_tokens 4096 \
--max_epochs 6 \
--lr 6e-6 \
--max_grad_norm 1 \
--warmup_ratio -1 \
--vocab_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/vocab/vocab \
--config_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/bert_large.json \
--save_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large/srl/conll09/6e-6 \
--predict \
--save \
--test_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009 \
--tensorboard \
--init_checkpoint /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large_dp=0.1_attdp=0.1_lr=6e-4_step=30000/checkpoints/ckpt_33000.pt \
--frames_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009/frames.json \
--dataset_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009/${task}.english.conll09.json \
--output_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/srl/conll2009/large/${task}.english.psense.conll09.json \
--checkpoint_path /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large/srl/conll09/6e-6 \
--tqdm_mininterval 1
done
