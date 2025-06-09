# 驱逐部分
# bash /home/avnet/xiongjing/sjh/parallel_window/run_scripts/head2.sh
# bash /home/avnet/xiongjing/sjh/parallel_window/run_scripts/head1.sh
# bash /home/avnet/xiongjing/sjh/parallel_window/run_scripts/head3.sh
# bash /home/avnet/xiongjing/sjh/parallel_window/run_scripts/head4.sh
export THRES=0.1
# infinite bench 一个窗口
# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
#     --in_eager_mode False --calibration_stage None  --calibration_mode 0

# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_3_6  --calibration_mode 1

# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_0_7  --calibration_mode 1

# # infinite bench 一个窗口
# bash run_test_infinitebench_multi_gpu_window8.sh \
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

# bash run_test_infinitebench_multi_gpu_window8.sh \
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
#     --key_no_rope True --positional_sorting IDreuse --rank_ascend True \
#     \
#     --attn_avg False --input_stitching False \
#     --in_eager_mode False --calibration_stage None  --calibration_mode 0

# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_8_15  --calibration_mode 1

# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_16_23  --calibration_mode 1


# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
#     --in_eager_mode False --calibration_stage head2_prefill_2_calibration_24_31  --calibration_mode 1

# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
#     --in_eager_mode False --calibration_stage head1_prefill_2_calibration_0_7  --calibration_mode 1

# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     --special_token True \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt False \
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
#     --in_eager_mode False --calibration_stage head1_prefill_2_calibration_8_15  --calibration_mode 1
# 1
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
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_2_calibration_16_23  --calibration_mode 1

#2 head1
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
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_2_calibration_24_31  --calibration_mode 1

#3 驱逐middle
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
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head3_prefill_2_calibration_24_31  --calibration_mode 1

#4 多窗口 bottom3 + 驱逐
bash run_test_infinitebench_multi_gpu_window8.sh \
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
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_24_31  --calibration_mode 1

#5 bottom5
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -5 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None  --calibration_mode 0

#6 bottom1 stage_eviction 1024 + 驱逐
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction True --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8  --delete_context_prompt False \
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
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_24_31  --calibration_mode 1

#7 bottom1 stage_eviction 1024 + 驱逐 head1
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction True --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8  --delete_context_prompt False \
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
    --in_eager_mode False --calibration_stage head1_prefill_2_calibration_8_15  --calibration_mode 1

#8 bottom1 stage_eviction 2048 + 驱逐
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction True --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt False \
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
    --in_eager_mode False --calibration_stage head2_prefill_2_calibration_24_31  --calibration_mode 1

#9 special_token 为 false head1_prefill_2_calibration_8_15
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
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_2_calibration_8_15  --calibration_mode 1

#10 bottom3 idreuse
bash run_test_infinitebench_multi_gpu_window8.sh \
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
    --key_no_rope True --positional_sorting IDreuse --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None  --calibration_mode 0

#11 多窗口 bottom3 + 驱逐 head1
bash run_test_infinitebench_multi_gpu_window8.sh \
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
    --in_eager_mode False --calibration_stage head1_prefill_2_calibration_24_31  --calibration_mode 1

#12 bottom1 stage_eviction 1024
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction True --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8  --delete_context_prompt False \
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
    --in_eager_mode False --calibration_stage None  --calibration_mode 0

#13 bottom1 stage_eviction 2048 
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction True --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt False \
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
    --in_eager_mode False --calibration_stage None  --calibration_mode 0