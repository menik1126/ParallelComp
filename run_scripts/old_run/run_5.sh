# # 1、进一步分解窗口 loss降序 decomposition_factor 3
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache True \
#     --window_ascend False  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 3 \

# # 2、loss升序 decomposition_factor 3
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache True \
#     --window_ascend True  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 3 \

# # 3、loss乱序 decomposition_factor 3
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage_random \
#     --gpu_nums 8  --truncation_method WO_Truncation \
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
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 3 \
# # 4、loss 升序 decomposition_factor 5
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache True \
#     --window_ascend True  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 5 \
# # 5、 loss 乱序 decomposition_factor 5
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage_random \
#     --gpu_nums 8  --truncation_method WO_Truncation \
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
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 5 \
# # 6、loss 升序 decomposition_factor 10
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache True \
#     --window_ascend True  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 10 \
# # 7、loss 乱序 decomposition_factor 10
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage_random \
#     --gpu_nums 8  --truncation_method WO_Truncation \
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
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 10 \
# # 8、loss 升序 decomposition_factor 15
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache True \
#     --window_ascend True  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 15 \
# # 9、loss 乱序 decomposition_factor 10
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage_random \
#     --gpu_nums 8  --truncation_method WO_Truncation \
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
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 15 \

# # 10、loss 升序 decomposition_factor 20
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache True \
#     --window_ascend True  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 20 \

# # 11、loss 乱序 decomposition_factor 20
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage_random \
#     --gpu_nums 8  --truncation_method WO_Truncation \
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
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 20 \

# # 12、loss 升序 decomposition_factor 40
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 10 --rank_cache True \
#     --window_ascend True  --reduce_length False --query_rank False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 40 \

# # 13、loss 乱序 decomposition_factor 40
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage_random \
#     --gpu_nums 8  --truncation_method WO_Truncation \
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
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 40 \


# # 14、随机保留窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage_shuffle \
#     --gpu_nums 8  --truncation_method WO_Truncation \
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
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 1 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 3 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows -1 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --get_recent_attn True \
    \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8  --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows -3 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# # 3
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 1 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 10  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# # 4
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows -1 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 10  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# # 5
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 10  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# 7
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 3 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 100  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# 8
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 3 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank True \
#     --recent_top_num 0  --get_recent_attn False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# 6 注意力筛选 10 筛选bottom 5个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows -5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 10  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# # 9 query loss筛选 bottom3
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows -3 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank True \
#     --recent_top_num 0  --get_recent_attn False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# # 1 query loss筛选 top5
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank True \
#     --recent_top_num 0  --get_recent_attn False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# 2 query loss筛选 bottom5
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows -5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank True \
#     --recent_top_num 0  --get_recent_attn False \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
    --gpu_nums 8 --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
    --NTK_aware False --special_token True \
    \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False --raw_position_select False \
    \
    --dynamic_window True --del_val False \
    --rank_windows True --topk_windows -1 --rank_cache False \
    --window_ascend False  --reduce_length False --query_rank True \
    --recent_top_num 0  --get_recent_attn False \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# 3 注意力筛选 100 筛选top 5个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 100  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# 4 注意力筛选 100 筛选bottom 5个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows -5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 100  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

# # 5 累积注意力筛选 跨层跨头 100 筛选top 5个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num -100  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

# # 6 注意力筛选 10 筛选top 3个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 3 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 10  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# 7 注意力筛选 10 筛选bottom 3个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows -3 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 10  --get_recent_attn True \
    # \
    # --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \
# 8 注意力筛选 10 筛选top 5个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 10  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

# 9 注意力筛选 100 筛选top 3个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows 3 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num 100  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \

# 10 累积注意力筛选 跨层跨头 100 筛选bottom 5个窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 8 --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --attn_implementation flash_attention_2 \
#     --NTK_aware False --special_token True \
#     \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False --raw_position_select False \
#     \
#     --dynamic_window True --del_val False \
#     --rank_windows True --topk_windows -5 --rank_cache False \
#     --window_ascend False  --reduce_length False --query_rank False \
#     --recent_top_num -100  --get_recent_attn True \
#     \
#     --position_shift False --shift_factor 0 --interval_shift 0 --decomposition_factor 1 \