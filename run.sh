NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "Running training on $NUM_GPUS GPU(s)"
torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py