CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 --master_port $(( $RANDOM % 99000 + 1000 )) /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/run_qa.py --do_train --evaluate_during_training  --model_type bert --train_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/qa/hotpotqa/train.jsonl --predict_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/qa/hotpotqa/dev.jsonl --doc_stride 128 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 128 --learning_rate 3e-5 --weight_decay 0.01 --num_train_epochs 2 --max_seq_length 512 --save_steps 1143 --output_dir /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large/hotpotqa --config_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/bert_large.json --vocab_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/vocab/vocab --init_checkpoint /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large_dp=0.1_attdp=0.1_lr=1e-3_step=30000/checkpoints/ckpt_33000.pt --do_lower_case --overwrite_output_dir 2>&1 | tee -a /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large/hotpotqa/results.txt
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node 1 --master_port $(( $RANDOM % 99000 + 1000  )) /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/run_qa.py --do_eval --evaluate_during_training  --model_type bert --train_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/qa/hotpotqa/train.jsonl --predict_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/../data/download/qa/hotpotqa/test.jsonl --doc_stride 128 --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 128 --learning_rate 3e-5 --weight_decay 0.01 --num_train_epochs 2 --max_seq_length 512 --save_steps 1143 --output_dir /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large/hotpotqa --config_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/bert_large.json --vocab_file /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/vocab/vocab --init_checkpoint /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large_dp=0.1_attdp=0.1_lr=1e-3_step=30000/checkpoints/ckpt_33000.pt --do_lower_case --overwrite_output_dir 2>&1 | tee -a /apdcephfs/share_47076/weizechen/hilbert/residual-timecorrect/results/large/hotpotqa/test_result.txt