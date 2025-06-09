# bash run_test_longbench_multi_gpu_window8_llama2.sh \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
#     \
#     --topk_windows -3 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0

# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --parallel_pattern default_label_1 --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 3000 --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 24000 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0

# bash run_test_infinitebench_multi_gpu_window8_all.sh \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 3000 --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -4 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0

# bash run_test_infinitebench_multi_gpu_window8_all.sh \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 3000 --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0

# bash run_test_infinitebench_multi_gpu_window8_all.sh \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction True \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 3000 --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -4 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0


# bash run_test_longbench_multi_gpu_window8_gemma2.sh \
#     --model google/gemma-2-9b \
#     --parallel_pattern default --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 3000 --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0

# bash run_test_longbench_multi_gpu_window8_llama2.sh \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --parallel_pattern default_label_11000_1.2 --gpu_nums 6 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 3000 --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 11000 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction True \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank False --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

# bash run_test_longbench_multi_gpu_window8_llama2.sh \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --parallel_pattern default --gpu_nums 6 \
#     --special_token True \
#     \
#     --kv_cache_eviction True \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 3000 --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -4 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_3_6 --calibration_mode 11

# bash run_test_longbench_multi_gpu_window8_gemma2.sh \
#     --model google/gemma-2-9b \
#     --parallel_pattern default --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2000  --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0

# bash run_test_longbench_multi_gpu_window8_gemma2.sh \
#     --model google/gemma-2-9b \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2000 --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -4 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0

# bash run_test_longbench_multi_gpu_window8_gemma2.sh \
#     --model google/gemma-2-9b \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2000 --recent_token 8  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0