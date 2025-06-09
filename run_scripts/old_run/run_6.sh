# # special token
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token False \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0

# # interval_shift
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 1

# # interval_shift 2
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 2

# # interval_shift 3
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 3

# # interval_shift 4
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 4

# # interval_shift 5
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 5

# # interval_shift 6
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 6


# # position shift
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 7 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    --NTK_aware False --special_token True \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows 10 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank False \
    \
    --position_shift True --shift_factor -1 --interval_shift 0

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor -2 --interval_shift 0

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor -3 --interval_shift 0

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor -4 --interval_shift 0

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor -5 --interval_shift 0

# # 随机顺序
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage_random \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage_random \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0

# # interval_shift
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 1

# # interval_shift 2
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 2

# interval_shift 3
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 3

# interval_shift 4
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 4

# interval_shift 5
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 5

# interval_shift 6
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift True --shift_factor 0 --interval_shift 6