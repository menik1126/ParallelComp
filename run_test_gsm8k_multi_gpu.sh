#!/bin/bash
OUTPUT_DIR=output_complex_cot_pcw_test
Model=meta-llama/Llama-2-7b-chat-hf #meta-llama/Llama-2-7b-chat-hf    gpt2-xl
Prompt_Method=complex_cot_pcw_multi_windows  #complex_cot_pcw #complex_cot_pcw_pre_process_window_cache    #complex_cot_pcw      #other   #complex_cot
output_json="complex_cot_test_result_pcw_with_window_cache_context_task_text_together_forward_delete_enter_last_lets_multigpu.json"
model_class="modeling_llama_with_pcw"
gpu=8
#export ROCR_VISIBLE_DEVICES=1,2
port=5327

accelerate launch  --main_process_port ${port} --num_processes ${gpu} run_evaluation_multi_gpu.py --dataset gsm8k  --n-windows 2  --subsample-test-set 2000 --n-runs 1 --output-dir $OUTPUT_DIR --model $Model --prompt_method $Prompt_Method --output_json $output_json --model_class $model_class 