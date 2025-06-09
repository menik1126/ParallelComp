# llama3 8B
# bash run_test_longbench_multi_gpu_window8.sh \
    # --parallel_pattern default --gpu_nums 8 \
    # --special_token True \
    # \
    # --kv_cache_eviction False \
    # --stage_eviction False --kv_cache_dynamic False \
    # --capacity 2048 --recent_token 1  --delete_context_prompt False \
    # \
    # --topk_windows -1 \
    # --reduce_length False --reduce_factor 0 \
    # --get_recent_attn False  --recent_top_num 0 \
    # --query_rank False --query_recent_tokens 0 \
    # \
    # --position_shift False --shift_factor 0 --interval_shift 0 \
    # --key_no_rope False --positional_sorting None --rank_ascend True \
    # \
    # --attn_avg False --input_stitching False \
    # --in_eager_mode False --calibration_stage None  --calibration_mode 0

# bash run_test_longbench_multi_gpu_window8.sh \
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
# llama2
# NTK
# bash run_test_longbench_multi_gpu_window8_test_llama2.sh \
#     --parallel_pattern default_NTK --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
# # PI
# bash run_test_longbench_multi_gpu_window8_llama2.sh \
#     --parallel_pattern default_PI --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate chunkllama
# # chunkllama
# bash run_test_longbench_multi_gpu_window8_llama2.sh \
#     --parallel_pattern default_chunkllama --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting chunkllama --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0
# conda activate parallel
# # 校准实验
# bash run_test_longbench_multi_gpu_window8_llama2.sh \
#     --parallel_pattern defalut_calibration --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg True --input_stitching False \
#     --in_eager_mode True --calibration_stage None  --calibration_mode 2
# llama3
# # NTK
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern default_NTK --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
# PI
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern default_PI --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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

# source ~/miniconda3/etc/profile.d/conda.sh 
# conda activate chunkllama
# # chunkllama
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern default_chunkllama --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting chunkllama --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0
# conda activate parallel

# # 校准实验
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern defalut_calibration --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg True --input_stitching False \
#     --in_eager_mode True --calibration_stage None  --calibration_mode 2

# bash run_test_longbench_multi_gpu_window8.sh \
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
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_24_27 --calibration_mode 1

bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -2 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -2 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_24_24 --calibration_mode 1

# # llama3 infinitebench
# # # NTK
# bash run_test_infinitebench_multi_gpu_window8_all.sh \
#     --parallel_pattern default_NTK --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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

# # PI
# bash run_test_infinitebench_multi_gpu_window8_all.sh \
#     --parallel_pattern default_PI --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
# source ~/miniconda3/etc/profile.d/conda.sh 
# conda activate chunkllama
# # chunkllama
# bash run_test_infinitebench_multi_gpu_window8_all.sh \
#     --parallel_pattern default_chunkllama --gpu_nums 7 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting chunkllama --rank_ascend True \
#     \
#     --attn_avg False --input_stitching True \
#     --in_eager_mode False --calibration_stage None --calibration_mode 0
# conda activate parallel
# # 校准实验
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern defalut_calibration --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
#     \
#     --topk_windows -1 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True \
#     \
#     --attn_avg True --input_stitching False \
#     --in_eager_mode True --calibration_stage None  --calibration_mode 2

# ours
bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_infinitebench_multi_gpu_window8_retrieval.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -1 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_24_31 --calibration_mode 1

bash run_test_infinitebench_multi_gpu_window8_retrieval.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -1 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_24_31 --calibration_mode 1

bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_16_23 --calibration_mode 1

bash run_test_infinitebench_multi_gpu_window8_retrieval.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -1 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_16_23 --calibration_mode 1

bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -2 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0