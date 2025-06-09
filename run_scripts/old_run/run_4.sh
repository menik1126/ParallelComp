# ## 0、NTK_aware 完全不再截断 
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern default \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window False --del_val False \
#     --rank_windows False --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0

# ## 0、NTK_aware 完全不再截断 
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern default \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window False --del_val False \
#     --rank_windows False --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0

# ## 1、NTK_aware 完全不再截断 
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window False --del_val False \
#     --rank_windows False --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0

# ## 2、NTK_aware 完全不再截断
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window False --del_val False \
#     --rank_windows False --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0

# ## 3、动态窗口下的NTK_aware
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0

## 3、动态窗口下的NTK_aware
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0

# ## 4、动态窗口下的NTK_aware
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0

# ## 5、动态窗口下的NTK_aware query_rank
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank True \
#     \
#     --position_shift False --shift_factor 0