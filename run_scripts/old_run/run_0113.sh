# 1、减少窗口到3000，查看效果
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8 \
    \
    --topk_windows -3 \
    --reduce_length True --reduce_factor 3000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True
    

# 2、减少窗口到3000，查看效果
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8 \
    \
    --topk_windows -1 \
    --reduce_length True --reduce_factor 3000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True
    

# 3、减少窗口到3000，查看效果
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8 \
    \
    --topk_windows -6 \
    --reduce_length True --reduce_factor 3000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False \
    --positional_sorting None --rank_ascend True

# 4、减少窗口到2000，查看效果
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8 \
    \
    --topk_windows -3 \
    --reduce_length True --reduce_factor 2000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True
    

# 5、减少窗口到2000，查看效果
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8 \
    \
    --topk_windows -1 \
    --reduce_length True --reduce_factor 2000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True
    

# 6、减少窗口到2000，查看效果
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8 \
    \
    --topk_windows -6 \
    --reduce_length True --reduce_factor 2000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True

# wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong
# 7、跨层比较recent 1token的top1注意力分数 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 1 \
    \
    --topk_windows 1 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn True  --recent_top_num -1 \
    --query_rank False --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True

# 8、跨层比较recent 8token的top1注意力分数 
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 8 \
    \
    --topk_windows 1 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn True  --recent_top_num -1 \
    --query_rank False --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True
    

# 9、跨层比较recent 1token的top1注意力分数 窗口为2000的实验
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 1024 --recent_token 1 \
    \
    --topk_windows 1 \
    --reduce_length True --reduce_factor 2000 \
    --get_recent_attn True  --recent_top_num -1 \
    --query_rank False --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True

# 10、6个窗口 kv size为2048，查看性能
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction True \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -6 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True
    

# 11、6个窗口 context为2048，PI 排序
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -6 \
    --reduce_length True --reduce_factor 2000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope True --positional_sorting PI --rank_ascend True
    

# 12、6个窗口 kv size为2048 PI 排序
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction True \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -6 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope True --positional_sorting PI --rank_ascend True
# wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong wrong    


# 13、3个窗口 offset 10*rank
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 10 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True

# 14、3个窗口 offset 右对齐
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 4000 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True

# 15、随机3个窗口 对照试验（筛选方式判定）
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new_shuffle --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank False --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True

# 16、2000长度窗口 query_rank bottom3 排序：IDoffset 偏移10*rank个单位
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new_shuffle --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -3 \
    --reduce_length True --reduce_factor 2000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 10 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True

# 17、2000长度窗口 query_rank bottom6 排序：IDoffset 偏移10*rank个单位
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new_shuffle --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -6 \
    --reduce_length True --reduce_factor 2000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 10 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True


# 18、2000长度窗口 query_rank bottom6 排序：IDoffset 偏移10*rank个单位
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new_shuffle --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -3 \
    --reduce_length True --reduce_factor 2000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 4000 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True


# 19、2000长度窗口 query_rank bottom6 排序：IDoffset 偏移10*rank个单位
bash run_test_longbench_multi_gpu_window8.sh \
    --parallel_pattern every_window_query_input_query_anchor_new_shuffle --gpu_nums 8 \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 8 \
    \
    --topk_windows -6 \
    --reduce_length True --reduce_factor 2000 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 4000 --interval_shift 0 \
    --key_no_rope True --positional_sorting IDoffset --rank_ascend True