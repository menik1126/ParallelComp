#!/bin/bash
task=gsm8k
response_file="/home/xiongjing/sjh/parallel_window_size/test1.json"
# pred_RGER_RGER_rerank32_kernel_method_sp_new_bs16_perdevice_4_lr_1e-6.json
file_name=$(basename ${response_file})
echo file_name: ${file_name}
output_file=results/${task}/${file_name}
python evaluate_results.py --task ${task} --response_file ${response_file} --output_file ${output_file}
