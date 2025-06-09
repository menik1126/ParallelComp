# 6、其他数据集 prefill_2 trieval外
# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --in_eager_mode False --calibration_stage None  --calibration_mode 0

# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --in_eager_mode False --calibration_stage head1_prefill_2_calibration_8_15  --calibration_mode 1

# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --in_eager_mode False --calibration_stage head1_prefill_2_calibration_16_23  --calibration_mode 1

# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_24_31  --calibration_mode 1

# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_16_23  --calibration_mode 1

# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_8_15  --calibration_mode 1

# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_0_7  --calibration_mode 1


###########################
# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_0_16  --calibration_mode 1

# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --attn_avg True --input_stitching False \
#     --in_eager_mode True --calibration_stage None  --calibration_mode 2

# llama2相关
# # 2、 infinitebench  NTK
# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --key_no_rope True --positional_sorting NTK --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None  --calibration_mode 0

# # 3、 infinitebench PI
# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --key_no_rope True --positional_sorting PI --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None  --calibration_mode 0

# # 4、 infinitebench 校准实验
# bash run_test_infinitebench_multi_gpu_window8_temp.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --attn_avg True --input_stitching False \
#     --in_eager_mode True --calibration_stage None  --calibration_mode 2


#!/bin/bash
# 1、 default 1
# chunk llama infinitebench
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate chunkllama
# 4、 chunkllama 6
bash run_test_infinitebench_multi_gpu_window8_temp_trieval.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope False --positional_sorting chunkllama --rank_ascend True \
    \
    --attn_avg False --input_stitching True \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_infinitebench_multi_gpu_window8_temp.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope False --positional_sorting chunkllama --rank_ascend True \
    \
    --attn_avg False --input_stitching True \
    --in_eager_mode False --calibration_stage None --calibration_mode 0
conda activate parallel
# llama3 8B
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern default --gpu_nums 8 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -1 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank False --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None  --calibration_mode 0
# 6、 ours  2
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
# 2、 PI 3
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope True --positional_sorting PI --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0
# 3、 NTK 4
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope True --positional_sorting NTK --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

# 5、 校准实验 5
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --attn_avg True --input_stitching False \
    --in_eager_mode True --calibration_stage None  --calibration_mode 2
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate chunkllama
# 4、 chunkllama 6
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope False --positional_sorting chunkllama --rank_ascend True \
    \
    --attn_avg False --input_stitching True \
    --in_eager_mode False --calibration_stage None --calibration_mode 0
conda activate parallel

# 7、OURS-eviction 7 8
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_3_6 --calibration_mode 1

bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_0_16 --calibration_mode 1

# infinitebench 相关实验
# 1、 infinitebench NTK 9
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope True --positional_sorting NTK --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None  --calibration_mode 0

# 1、 infinitebench  NTK 10
bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope True --positional_sorting NTK --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None  --calibration_mode 0

# 2、 infinitebench PI 11
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope True --positional_sorting PI --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None  --calibration_mode 0

# 2、 infinitebench  PI 12
bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope True --positional_sorting PI --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None  --calibration_mode 0

# 3、校准实验 infinitebench 13
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --attn_avg True --input_stitching False \
    --in_eager_mode True --calibration_stage None  --calibration_mode 2

# 3、校准实验 infinitebench 14
bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --attn_avg True --input_stitching False \
    --in_eager_mode True --calibration_stage None  --calibration_mode 2

# 4、ours 15
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --attn_avg True --input_stitching False \
    --in_eager_mode True --calibration_stage None  --calibration_mode 0

# 4、ours 16
bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --attn_avg True --input_stitching False \
    --in_eager_mode True --calibration_stage None  --calibration_mode 0

# 5、ours-eviction 17
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
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
    --attn_avg True --input_stitching False \
    --in_eager_mode True --calibration_stage head1_prefill_2_calibration_16_23  --calibration_mode 1

# 5、ours-eviction 18
bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --attn_avg True --input_stitching False \
    --in_eager_mode True --calibration_stage head2_prefill_2_calibration_0_16  --calibration_mode 1

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate chunkllama
# 6、chunk llama 19 20
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
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
    --key_no_rope False --positional_sorting chunkllama --rank_ascend True \
    \
    --attn_avg False --input_stitching True \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_infinitebench_multi_gpu_window8_other.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
    --key_no_rope False --positional_sorting chunkllama --rank_ascend True \
    \
    --attn_avg False --input_stitching True \
    --in_eager_mode False --calibration_stage None --calibration_mode 0
conda activate parallel