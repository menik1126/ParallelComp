#!/bin/bash
OUTPUT_DIR=output_complex_cot_pcw_test
Model=/mnt/Data/xiongjing/llama2chat #meta-llama/Llama-3.1-8B-Instruct #meta-llama/Llama-2-7b-chat-hf #meta-llama/Llama-2-7b-chat-hf    gpt2-xl
Prompt_Method=complex_cot_pcw_multi_windows_kv_cache  #complex_cot_pcw #complex_cot_pcw_pre_process_window_cache    #complex_cot_pcw      #other   #complex_cot
output_json="test_kv_cache.json"
sample_method="8-shot"   #"8-shot"  #"sample"  #8-shot
sample_number=16
extra_sample_number=0
model_class="modeling_llama_with_pcw_kv_cache"
gpu=3
port=5327
# export CUDA_VISIBLE_DEVICES=3
export NCCL_P2P_DISABLE=1
config_files=scripts/gpu_3.yaml
accelerate launch --config_file $config_files --main_process_port ${port} --num_processes ${gpu} \
    run_evaluation_multi_gpu.py \
    --dataset gsm8k  --n-windows 2  --subsample-test-set 2000 --n-runs 1 --output-dir $OUTPUT_DIR \
    --model $Model --prompt_method $Prompt_Method --output_json $output_json --model_class $model_class  \
    --sample_number $sample_number --sample_method $sample_method