# # 1 kv_cache stage压缩
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction True --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False \
#     --positional_sorting None --rank_ascend True

# # 2 kv_cache 动态压缩 stage压缩
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction True --kv_cache_dynamic True \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False \
#     --positional_sorting None --rank_ascend True

# # 3 NTK 排序 升序
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting NTK --rank_ascend True
    
# 4 NTK 排序 降序
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting NTK --rank_ascend False

# # 5 PI 排序 升序
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting PI --rank_ascend True

# # 6 PI 排序 降序
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting PI --rank_ascend False

# 7 IDreuse 排序 升序
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting IDreuse --rank_ascend True

# # 8 IDreuse 排序 降序
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting IDreuse --rank_ascend False

# # 9 IDoffset 排序 升序
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting IDoffset --rank_ascend True

# # 10 IDoffset 排序 降序
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope True --positional_sorting IDoffset --rank_ascend False

# # 11 last1注意力分数筛选  top10
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 1 \
#     \
#     --topk_windows 3 \
#     --reduce_length False \
#     --get_recent_attn True  --recent_top_num 10 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False \
#     --positional_sorting None --rank_ascend True

# # 12 last1注意力分数筛选  top1
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 1 \
#     \
#     --topk_windows 3 \
#     --reduce_length False \
#     --get_recent_attn True  --recent_top_num 1 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False \
#     --positional_sorting None --rank_ascend True

# 13 kv 驱逐 2k
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction True \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend False

# 14 kv 驱逐 2k stage驱逐
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction True --kv_cache_dynamic False \
#     --capacity 2048 --recent_token 8 \
#     \
#     --topk_windows -3 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False --positional_sorting None --rank_ascend False

# # 15 last1注意力分数筛选  top10
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 1 \
#     \
#     --topk_windows 3 \
#     --reduce_length False \
#     --get_recent_attn True  --recent_top_num -10 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False \
#     --positional_sorting None --rank_ascend True

# # 16 last1注意力分数筛选  top1
# bash run_test_longbench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 1 \
#     \
#     --topk_windows 3 \
#     --reduce_length False \
#     --get_recent_attn True  --recent_top_num -1 \
#     --query_rank False --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False \
#     --positional_sorting None --rank_ascend True

# 17 infinitebench query_loss bottom3
bash run_test_infinitebench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8 \
    \
    --topk_windows -3 \
    --reduce_length False \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False \
    --positional_sorting None --rank_ascend True

# # 18 infinitebench query_loss bottom10
# bash run_test_infinitebench_multi_gpu_window8.sh \
#     --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 7 \
#     \
#     --kv_cache_eviction False \
#     --stage_eviction False --kv_cache_dynamic False \
#     --capacity 1024 --recent_token 8 \
#     \
#     --topk_windows -10 \
#     --reduce_length False \
#     --get_recent_attn False  --recent_top_num 0 \
#     --query_rank True --query_recent_tokens 0 \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 \
#     --key_no_rope False \
#     --positional_sorting None --rank_ascend True