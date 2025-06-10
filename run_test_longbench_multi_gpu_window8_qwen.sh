#!/bin/bash

# export NCCL_TIMEOUT=6000
# export RCCL_TIMEOUT=6000
# export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_SOCKET_TIMEOUT=6000
# export NCCL_ASYNC_ERROR_HANDLING=1 

#用到的参数如下：
Model="Qwen/Qwen2.5-7B-Instruct" #
parallel_pattern="every_window_query_input_query" #"default"
n_windows=2
topk_windows=10
# default

gpu=5
# 默认参数
model_class="modeling_qwen2_with_pcw_kv_cache_FlashAttention_longbench"
special_token=True
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) Model="$2"; shift ;;
        --gpu_nums) gpu="$2"; shift ;;
        --parallel_pattern) parallel_pattern="$2"; shift ;;
        --kv_cache_eviction) kv_cache_eviction=(${2//,/ }); shift ;;
        --capacity) capacity="$2"; shift ;;
        --kv_cache_dynamic) kv_cache_dynamic="$2"; shift ;;
        --recent_token) recent_token="$2"; shift ;;
        --topk_windows) topk_windows="$2"; shift ;;
        --stage_eviction) stage_eviction="$2"; shift ;;
        --reduce_length) reduce_length="$2"; shift ;;
        --query_rank) query_rank="$2"; shift ;;
        --special_token) special_token="$2"; shift ;;
        --query_recent_tokens) query_recent_tokens="$2"; shift ;;
        --reduce_factor) reduce_factor="$2"; shift ;;
        --calibration_stage) calibration_stage="$2"; shift ;;
        --calibration_mode) calibration_mode="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$parallel_pattern" == *"default"* ]]; then
    n_windows=1
    echo "n_windows set to $n_windows"
fi

port=5328
config_files=scripts/gpu_$gpu.yaml
datasets=("narrativeqa" 
         "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" \
          "trec" "triviaqa" "passage_count" "passage_retrieval_en" \
          "qmsum" "samsum" "lcc"   "multi_news" "repobench-p" "gov_report" )

for dataset in ${datasets[@]};do
    echo "Running evaluation for dataset: $dataset"
    accelerate launch --main_process_port ${port} --num_processes ${gpu}  --config_file ${config_files} \
        run_evaluation_multi_gpu.py \
        --n-windows $n_windows \
        \
        --model_class $model_class \
        --model $Model --dataset $dataset  \
        --parallel_pattern $parallel_pattern \
        --special_token $special_token \
        \
        --kv_cache_eviction $kv_cache_eviction \
        --capacity $capacity --kv_cache_dynamic $kv_cache_dynamic --recent_token $recent_token \
        --stage_eviction $stage_eviction \
        \
        --topk_windows $topk_windows --reduce_factor $reduce_factor \
        --query_rank $query_rank --query_recent_tokens $query_recent_tokens \
        \
        --calibration_stage $calibration_stage --calibration_mode $calibration_mode
done
