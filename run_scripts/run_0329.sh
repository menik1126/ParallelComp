### llama3.1 
bash run_test_longbench_multi_gpu_window8_qwen2.sh \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --parallel_pattern default_label_24000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -1 \
    --reduce_length False --reduce_factor 24000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank False --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_longbench_multi_gpu_window8_qwen2.sh \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new_label_6000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 24000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_longbench_multi_gpu_window8_qwen2.sh \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new_label_6000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction True \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 24000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

## QWEN2
bash run_test_longbench_multi_gpu_window8_qwen2.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new_label_6000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 6500 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_3_6 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_qwen2.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new_label_6000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction True \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 6500 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_longbench_multi_gpu_window8_qwen2.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new_label_6000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction True \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 6500 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_3_6 --calibration_mode 11
# infinitebench 

bash run_test_infinitebench_multi_gpu_window8_qwen2.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parallel_pattern default_label_24000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -1 \
    --reduce_length False --reduce_factor 24000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_infinitebench_multi_gpu_window8_qwen2.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parallel_pattern default_label_8000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -1 \
    --reduce_length False --reduce_factor 8000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_infinitebench_multi_gpu_window8_qwen2.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new_label_6000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 6500 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_infinitebench_multi_gpu_window8_qwen2.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new_label_6000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 6500 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_3_6 --calibration_mode 11

bash run_test_infinitebench_multi_gpu_window8_qwen2.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new_label_6000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction True \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 6500 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_infinitebench_multi_gpu_window8_qwen2.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parallel_pattern every_window_query_input_query_anchor_new_label_6000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction True \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 6500 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_3_6 --calibration_mode 11

## Gamma2
bash run_test_longbench_multi_gpu_window8_gemma2.sh \
    --model google/gemma-2-9b \
    --parallel_pattern default_label_4000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 4000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank False --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_longbench_multi_gpu_window8_gemma2.sh \
    --model google/gemma-2-9b \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 4000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0
### gemma2
bash run_test_longbench_multi_gpu_window8_gemma2.sh \
    --model google/gemma-2-9b \
    --parallel_pattern default_label_4000 --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 4000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank False --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_longbench_multi_gpu_window8_gemma2.sh \
    --model google/gemma-2-9b \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 6 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 3000 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -4 \
    --reduce_length False --reduce_factor 4000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0