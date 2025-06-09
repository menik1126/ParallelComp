# # 11、NTK排序 delete_context_prompt True
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 8  --delete_context_prompt True \
#     \
#     --topk_windows -3 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting NTK --rank_ascend True

# # 12、PI排序 delete_context_prompt True
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 8  --delete_context_prompt True \
#     \
#     --topk_windows -3 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting PI --rank_ascend True

# 14、窗口内排序 1000size delete_context_prompt true
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt True \
#     \
#     --topk_windows -3 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 1000 --interval_shift 0 \
#     --key_no_rope True --positional_sorting INwindow --rank_ascend True


# # 15、窗口内排序 2000size
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
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
#     --position_shift False --shift_factor 2000 --interval_shift 0 \
#     --key_no_rope True --positional_sorting INwindow --rank_ascend True

# # 16、窗口内排序 2000size delete_context_prompt true
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 1  --delete_context_prompt True \
#     \
#     --topk_windows -3 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 2000 --interval_shift 0 \
#     --key_no_rope True --positional_sorting INwindow --rank_ascend True

# # 17、随机筛选4个窗口
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new_shuffle --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 8 --delete_context_prompt False \
#     \
#     --topk_windows -4 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True

# # 18、query 筛选4个窗口 4*4k
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 8 --delete_context_prompt False \
#     \
#     --topk_windows -4 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend True

# # 19、NTK排序 delete_context_prompt True
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 8  --delete_context_prompt True \
#     \
#     --topk_windows -3 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting NTK --rank_ascend True

# # 20、PI排序 delete_context_prompt True
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 8  --delete_context_prompt True \
#     \
#     --topk_windows -3 \
#     --reduce_length False --reduce_factor 0 \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting PI --rank_ascend True

# 21、IDoffset delete_context_prompt True 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 15 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True \
    \
    --attn_implementation flash_attention_2 --attn_avg False

# 22、IDoffset delete_context_prompt True 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 20 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True \
    \
    --attn_implementation flash_attention_2 --attn_avg False

# 23、IDoffset delete_context_prompt True 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 10 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True \
    \
    --attn_implementation flash_attention_2 --attn_avg False


# 24、IDoffset delete_context_prompt True 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 5 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True \
    \
    --attn_implementation flash_attention_2 --attn_avg False

# 25、IDoffset delete_context_prompt True 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 15 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True \
    \
    --attn_implementation flash_attention_2 --attn_avg False

# 26、IDoffset delete_context_prompt True 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 20 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True \
    \
    --attn_implementation flash_attention_2 --attn_avg False

# 27、IDoffset delete_context_prompt True 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 10 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True \
    \
    --attn_implementation flash_attention_2 --attn_avg False


# 28、IDoffset delete_context_prompt True 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 5 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True \
    \
    --attn_implementation flash_attention_2 --attn_avg False

# 29、attn_calibration eager
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope True --positional_sorting None --rank_ascend True \
    \
    --attn_implementation eager --attn_avg False

# 30、attn_calibration attn_avg 为true
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope True --positional_sorting None --rank_ascend True \
    \
    --attn_implementation eager --attn_avg True