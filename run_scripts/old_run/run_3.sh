# ## 1 对比 位置偏移 偏移0位
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0
# ## 2 对比 位置偏移 偏移1位
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 1
# # 3 位置偏移 偏移2位
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 2

# 4 拉低context length 扩大为10个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length True --query_rank False \
#     \
#     --position_shift False --shift_factor 0

#5 top10 windows 升序排列 rank最高的在最recent
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache True \
#     --window_ascend True  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0

# # 6 拉低context length
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length True --query_rank False \
#     \
#     --position_shift False --shift_factor 0

# 7、query_rank
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank True \
#     \
#     --position_shift False --shift_factor 0

# 8、 位置偏移 偏移4位
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 4

# 9、 位置偏移 偏移8位
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 8