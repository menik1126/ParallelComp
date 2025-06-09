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