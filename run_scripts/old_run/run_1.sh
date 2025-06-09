# 1 base1
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query \
#     --gpu_nums 7 \
#     --raw_position_select False --truncation_method WO_Truncation --kv_cache_eviction False --capacity 1024 \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench --kv_cache_dynamic False \
#     --recent_token 8 --attn_implementation flash_attention_2 \
#     --del_val False --rank_windows False --topk_windows 10 --rank_cache False --stage_eviction False

# 2 # 添加query再删query 不驱逐
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --kv_cache_dynamic False --capacity 1024 --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows False --topk_windows 10 --rank_cache False

# 3 # 添加query再删query，query驱逐
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction True --kv_cache_dynamic False --capacity 1024 --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows False --topk_windows 10 --rank_cache False

# 4 # 添加query再删query，动态驱逐
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction True --capacity 1024 --kv_cache_dynamic True --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows False --topk_windows 10 --rank_cache False 

# 5 # 第二次prefill驱逐 非动态驱逐 kv size为1024*window_num
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction True \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows False --topk_windows 10 --rank_cache False 

# 6 # 第二次prefill驱逐 动态驱逐 kv size为1024*window_num 
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic True --recent_token 8 \
#     --stage_eviction True \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows False --topk_windows 10 --rank_cache False


# 7 窗口排序+驱逐
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache True

# 8 窗口驱逐+原有顺序
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False

# 9 窗口驱逐+原有顺序+kv驱逐
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction True --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache True

# 10 窗口驱逐+原有顺序+kv动态驱逐
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction True --capacity 1024 --kv_cache_dynamic True --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache True

# 11 窗口驱逐+bottom5
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_query_input_query_anchor_new \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows -5 --rank_cache False

# 12、动态窗口
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False --dynamic_window True

# # 13、动态窗口+最大值在前面 top
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache True --dynamic_window True

# # 14、动态窗口+最大值在前面 bottom
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows -5 --rank_cache True --dynamic_window True

# # 15、动态窗口+最大值在前面 top 动态驱逐
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction True \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False --dynamic_window True


# 16 动态窗口 shift 2
bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
    --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
    --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
    --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
    --stage_eviction False \
    --attn_implementation flash_attention_2 --del_val False \
    --rank_windows True --topk_windows 5 --rank_cache False --dynamic_window True \
    --position_shift True --shift_factor 2 --window_ascend False

# # 17 动态窗口 shift 1
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False --dynamic_window True \
#     --position_shift True --shift_factor 1 --window_ascend False

# 18 动态窗口 shift 4
# bash run_test_longbench_multi_gpu_window8.sh --parallel_pattern every_window_no_query_input_query_stage \
#     --gpu_nums 7 --raw_position_select False --truncation_method WO_Truncation \
#     --model_class modeling_llama_with_pcw_kv_cache_FlashAttention_longbench \
#     --kv_cache_eviction False --capacity 1024 --kv_cache_dynamic False --recent_token 8 \
#     --stage_eviction False \
#     --attn_implementation flash_attention_2 --del_val False \
#     --rank_windows True --topk_windows 5 --rank_cache False --dynamic_window True \
#     --position_shift True --shift_factor 4 --window_ascend False