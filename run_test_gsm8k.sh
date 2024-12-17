#!/bin/bash
OUTPUT_DIR=output_complex_cot_pcw_test
Model=/mnt/Data/xiongjing/llama2chat #meta-llama/Llama-2-7b-chat-hf    gpt2-xl
Prompt_Method=complex_cot_pcw_multi_windows #complex_cot_pcw  #complex_cot_pcw      #complex_cot      #other   #complex_cot
output_json="complex_cot_test_result_pcw_wo_window_cache_test6.json"
model_class="modeling_llama_with_pcw" #"modeling_llama" #"modeling_llama_with_pcw_wo_max_pos"
n_shots_per_window=8
export ROCR_VISIBLE_DEVICES=0

python run_evaluation.py --dataset gsm8k  --n-windows 2  --subsample-test-set 2000 --n-runs 1 --output-dir $OUTPUT_DIR --model $Model --prompt_method $Prompt_Method --output_json $output_json --model_class $model_class --n-shots-per-window $n_shots_per_window