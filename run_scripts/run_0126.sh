# 驱逐策略的消融
# 驱逐 sink
export THRES=0.2
bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token True \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt False \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage None --calibration_mode 0

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_1_calibration_0_7 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_1_calibration_4_7 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_1_calibration_8_11 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_1_calibration_12_15 --calibration_mode 11


bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_1_calibration_16_19 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_1_calibration_20_23 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_1_calibration_24_27 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head2_prefill_1_calibration_28_31 --calibration_mode 11

# 驱逐策略的消融
# 驱逐 recent
bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_1_calibration_12_15 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_1_calibration_8_11 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_1_calibration_4_7 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_1_calibration_0_3 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_1_calibration_16_19 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_1_calibration_20_23 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_1_calibration_24_27 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head1_prefill_1_calibration_28_31 --calibration_mode 11


# 驱逐策略的消融
# 驱逐 middle
bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head3_prefill_1_calibration_0_3 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head3_prefill_1_calibration_4_7 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head3_prefill_1_calibration_8_11 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head3_prefill_1_calibration_12_15 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head3_prefill_1_calibration_16_19 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head3_prefill_1_calibration_20_23 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head3_prefill_1_calibration_24_27 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head3_prefill_1_calibration_28_31 --calibration_mode 11


# 驱逐策略的消融
# 全部驱逐
bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head4_prefill_1_calibration_0_3 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head4_prefill_1_calibration_4_7 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head4_prefill_1_calibration_8_11 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head4_prefill_1_calibration_12_15 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head4_prefill_1_calibration_16_19 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head4_prefill_1_calibration_20_23 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head4_prefill_1_calibration_24_27 --calibration_mode 11

bash run_test_longbench_multi_gpu_window8_llama2.sh \
    --parallel_pattern every_window_query_input_query_anchor_new --gpu_nums 8 \
    --special_token False \
    \
    --kv_cache_eviction False \
    --stage_eviction False --kv_cache_dynamic False \
    --capacity 2048 --recent_token 1  --delete_context_prompt True \
    \
    --topk_windows -3 \
    --reduce_length False --reduce_factor 0 \
    --get_recent_attn False  --recent_top_num 0 \
    --query_rank True --query_recent_tokens 0 \
    \
    --position_shift False --shift_factor 0 --interval_shift 0 \
    --key_no_rope False --positional_sorting None --rank_ascend True \
    \
    --attn_avg False --input_stitching False \
    --in_eager_mode False --calibration_stage head4_prefill_1_calibration_28_31 --calibration_mode 11