set -e

for task in bias_add_simple_9010 bias_add_simple_1090 bias_add_simple; do
    for do_train num_test in True 1000 False 1000; do
        NCCL_IB_DISABLE="1" NCCL_P2P_DISABLE="1" CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online python run.py \
            --seed=44 \
            --task=$task \
            --num_mix_train=1 \
            \
            \
            --num_train=10_000_000 \
            --num_test=$num_test \
            --hidden_size=384 \
            --num_attention_heads=6 \
            --num_layers=6 \
            --max_position_embeddings=1024 \
            \
            \
            --ignore_data_skip=True \
            --resume_from_checkpoint=True \
            --save_total_limit=1 \
            --run_name='' \
            --output_dir=out3 \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=1500 \
            --learning_rate=5e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs='{"num_stable_steps": 1200, "num_decay_steps": 150}' \
            --adam_beta2=0.98 \
            --adam_epsilon=1e-8 \
            --weight_decay=0.01 \
            --max_grad_norm=1 \
            --warmup_ratio=0.1 \
            --logging_steps=20 \
            --eval_strategy="steps" \
            --eval_steps=200 \
            --per_device_train_batch_size=200 \
            --per_device_eval_batch_size=100 \
            --gradient_accumulation_steps=32 \
            --include_inputs_for_metrics=True \
            --torch_compile=True \
            --bf16=False \
            --tf32=True
    done
done