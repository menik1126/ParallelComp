#!/bin/bash
task=gsm8k
response_file="/home/xiongjing/sjh/parallel_window_size/complex_cot_test_result_pcw_with_window_cache_context_task_text_together_forward_delete_enter_last_lets_multigpu_n-windows_1_random_select_prompt_16_the_answer_is.json"
# pred_RGER_RGER_rerank32_kernel_method_sp_new_bs16_perdevice_4_lr_1e-6.json
file_name=$(basename ${response_file})
echo file_name: ${file_name}
output_file=results/${task}/${file_name}
python evaluate_results.py --task ${task} --response_file ${response_file} --output_file ${output_file}
