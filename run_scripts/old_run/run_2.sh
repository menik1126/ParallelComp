# 1 动态窗口 升序
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache True \
    --window_ascend True  --reduce_length False --query_rank False \
    \
    --position_shift False --shift_factor 0
# 2 动态窗口 降序
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache True \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift False --shift_factor 0
## 13 对比 原始顺序
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift False --shift_factor 0

# 3 动态窗口 数量拉大 原始顺序
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 10 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift False --shift_factor 0

# 4 动态窗口 数量拉大 原始顺序
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 8 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift False --shift_factor 0

# 5 动态窗口 数量拉小 原始顺序
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 3 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift False --shift_factor 0

# 6 动态窗口 数量拉小 原始顺序 选择最大的保留
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 1 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift False --shift_factor 0

## 14 对比 位置偏移 偏移0位
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift True --shift_factor 0

bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift True --shift_factor 1
# 8 位置偏移 偏移2位
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift True --shift_factor 2

# 9 位置偏移 偏移4位
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift True --shift_factor 4

# 10 位置偏移 偏移8位
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift True --shift_factor 8

# 11 拉低context length
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache False \
    --window_ascend False  --reduce_length True --query_rank False \
    \
    --position_shift False --shift_factor 0

# 12 query进行rank
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank True \
    \
    --position_shift False --shift_factor 0