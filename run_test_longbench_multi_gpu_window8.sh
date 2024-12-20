#!/bin/bash
OUTPUT_DIR=output_complex_cot_pcw_test
output_json="results_jsons/test1.json"
sample_method="8-shot"   #"8-shot"  #"sample"  #8-shot
sample_number=0
extra_sample_number=0


Model=/mnt/Data/xiongjing/llama2chat #meta-llama/Llama-3.1-8B-Instruct #meta-llama/Llama-2-7b-chat-hf #meta-llama/Llama-2-7b-chat-hf    gpt2-xl
Prompt_Method=complex_cot_pcw_multi_windows  #complex_cot_pcw #complex_cot_pcw_pre_process_window_cache    #complex_cot_pcw      #other   #complex_cot
model_class="modeling_llama_with_pcw"
parallel_pattern="default" 
## every_window_query_input_query 
# every_window_no_query_input_query
## every_window_query_input_no_query
# default

gpu=1
gpu_ids=3
port=5326
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export NCCL_P2P_DISABLE=1
config_files=scripts/gpu_1_$gpu_ids.yaml

datasets=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" \
          "trec" "triviaqa" "passage_count" "passage_retrieval_en" \
          "qmsum" "samsum" "lcc" "repobench-p" "gov_report" "multi_news")

# unset datasets
# datasets="multifieldqa_en"

for dataset in ${datasets[@]};do
    echo "Running evaluation for dataset: $dataset"
    accelerate launch --main_process_port ${port} --num_processes ${gpu}  --config_file ${config_files} \
        run_evaluation_longbench_multi_gpu.py \
        --dataset $dataset  --n-windows 1  --subsample-test-set 2000 --n-runs 1 --output-dir $OUTPUT_DIR \
        --model $Model --prompt_method $Prompt_Method --output_json $output_json --model_class $model_class \
        --sample_number $sample_number --sample_method $sample_method --extra_sample_number $extra_sample_number \
        --parallel_pattern $parallel_pattern
done