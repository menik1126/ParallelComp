#!/bin/bash
OUTPUT_DIR=output
Model=meta-llama/Llama-2-7b-chat-hf #meta-llama/Llama-2-7b-chat-hf    gpt2-xl
Prompt_Method=other   # complex_cot
export ROCR_VISIBLE_DEVICES=0

# python run_evaluation.py --dataset sst2  --n-windows 1 --n-windows 3  --subsample-test-set 250 --n-runs 30 --output-dir $OUTPUT_DIR --model $Model --model_class $Model_Class --prompt_method $Prompt_Method  > output.txt
python run_evaluation.py --dataset sst2 --n-windows 1 --n-windows 3 --subsample-test-set 250 --n-runs 30 --output-dir $OUTPUT_DIR --model $Model --prompt_method $Prompt_Method
