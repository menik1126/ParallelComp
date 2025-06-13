# ParallelComp: Parallel Long-Context Compressor for Length Extrapolation

[![arXiv](https://img.shields.io/badge/arXiv-2502.14317-b31b1b.svg)](https://arxiv.org/abs/2502.14317)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of **ParallelComp**, a novel training-free method for long-context extrapolation that extends Large Language Models' (LLMs) context length from 8K to 128K while maintaining high throughput and preserving perplexity.

## 📄 Paper

**ParallelComp: Parallel Long-Context Compressor for Length Extrapolation**  
*Jing Xiong, Jianghan Shen, Chuanyang Zheng, Zhongwei Wan, Chenyang Zhao, Chiwun Yang, Fanghua Ye, Hongxia Yang, Lingpeng Kong, Ngai Wong*

📖 [Paper Link](https://arxiv.org/abs/2502.14317)

## 🚀 Key Features

- **Training-free**: No costly fine-tuning required for length extrapolation
- **High Performance**: Achieves 91.17% of GPT-4's performance on long-context tasks using an 8B model
- **Scalable**: Extends context length from 4K to 128K tokens
- **Efficient**: Integrates seamlessly with Flash Attention
- **Fast**: 23.50x acceleration in the prefilling stage with 1.76x improvement in chunk throughput
- **Memory Efficient**: Manages ultra-long contexts on a single A100 80GB GPU

## 🛠️ Installation

### Requirements

- Python 3.9+
- PyTorch 2.5.1
- CUDA compatible GPU(s)
- Transformers 4.43.2
- 80GB A100 GPU memory for ultra-long contexts

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/ParallelComp.git
cd ParallelComp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For development, install from requirements.in:
```bash
pip-compile requirements.in
pip install -r requirements.txt
```

## 📊 Usage

### Quick Start

#### Single GPU Evaluation

```bash
bash run_test_longbench_multi_gpu_window8_llama.sh \
    --parallel_pattern parallel_comp --gpu_nums 1_0 \
    --kv_cache_eviction false --capacity 512 \
    --kv_cache_dynamic false --stage_eviction false \
    --recent_token 8  \
    --topk_windows -3 --query_rank true \
    --query_recent_tokens 0 --reduce_factor 0 \
    --calibration_stage None --calibration_mode 0 \
    --special_token true \
    --model meta-llama/Llama-2-7b-chat-hf
```

#### Multi-GPU Evaluation

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash run_test_longbench_multi_gpu_window8_llama.sh \
    --parallel_pattern parallel_comp --gpu_nums 4 \
    --kv_cache_eviction false --capacity 512 \
    --kv_cache_dynamic false --stage_eviction false \
    --recent_token 8  \
    --topk_windows -3 --query_rank true \
    --query_recent_tokens 0 --reduce_factor 0 \
    --calibration_stage None --calibration_mode 0 \
    --special_token true \
    --model meta-llama/Llama-2-7b-chat-hf \
```

#### Batch Evaluation

```bash
bash run_test_longbench_multi_gpu_window8_llama.sh \
    --parallel_pattern parallel_comp_batches --gpu_nums 1_0 \
    --kv_cache_eviction false --capacity 512 \
    --kv_cache_dynamic false --stage_eviction false \
    --recent_token 8  \
    --topk_windows -3 --query_rank true \
    --query_recent_tokens 0 --reduce_factor 0 \
    --calibration_stage None --calibration_mode 0 \
    --special_token true \
    --model meta-llama/Llama-2-7b-chat-hf
```

### Supported Models

- **LLaMA** family models
- **Qwen2.5** models

### Supported Datasets

#### Long Context Benchmarks
- **LongBench**: narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p
- **InfiniteBench**: passkey, number_string,kv_retrieval, math_find, code_debug, longbook_choice_eng, longdialogue_qa_eng


## 📈 Evaluation Scripts

### Evaluation Metrics

Calculate metrics for evaluation results:

```bash
# For LongBench results
bash scripts/longbench_metrics.sh --results_dir ./results --new_method your_method_name --switch true

# For InfiniteBench results  
bash scripts/infinitebench_metrics.sh --results_dir ./results --new_method your_method_name --switch true
```

### GPU Configuration

Multi-GPU setups use accelerate with YAML configs in `scripts/`:

```bash
# 4 GPU setup
accelerate launch --config_file scripts/gpu_4.yaml your_script.py

# 8 GPU setup  
accelerate launch --config_file scripts/gpu_8.yaml your_script.py
```

Available configurations: `gpu_1.yaml`, `gpu_2.yaml`, `gpu_3.yaml`, `gpu_4.yaml`, `gpu_5.yaml`, `gpu_6.yaml`, `gpu_7.yaml`, `gpu_8.yaml`

## 🔧 Advanced Configuration

### Attention Calibration

ParallelComp includes attention calibration strategies to mitigate attention sink issues:

#### Single GPU Evaluation

```bash
bash run_test_longbench_multi_gpu_window8_llama.sh \
    --parallel_pattern parallel_comp --gpu_nums 1_0 \
    --kv_cache_eviction false --capacity 512 \
    --kv_cache_dynamic false --stage_eviction false \
    --recent_token 8  \
    --topk_windows -3 --query_rank true \
    --query_recent_tokens 0 --reduce_factor 0 \
    --calibration_stage prefill_2_calibration_head_{sink/recent/middle/all}_{layer_i}_{layer_j} --calibration_mode 1 \
    --special_token true \
    --model meta-llama/Llama-2-7b-chat-hf \
```

## 🚧 TODO & Roadmap
- [❎] **Code Organization**: Currently organizing and cleaning up the codebase for better usability
- [❎] **Gemma Support**: Adding full support for Gemma model family
- [❎] **Baselines**: Adding full support for Evaluation of Baselines
- [❎] **SGLang Integration**: Adding support for SGLang inference engine for improved performance
- [❎] **Documentation**: Expanding documentation with more detailed examples
- [❎] **Quantization Support**: Adding support for model quantization (INT8/INT4) to reduce memory usage and accelerate inference
- [❎] **Benchmarks**: Adding more comprehensive benchmark results
- [✅] **FlashAttention Support**
- [✅] **Multi-GPU Inference Support**
- [✅] **Batch Inference Support**
- [✅] **AMD GPU Support**

## 📁 Project Structure

```
ParallelComp/
├── run_evaluation_multi_gpu.py                              # Multi-GPU evaluation script
├── model_loaders.py                                         # Model loading utilities
├── experiment_manager.py                                    # Experiment management
├── pcw_wrapper.py                                          # Parallel Context Window wrapper
├── modeling_llama_with_pcw_kv_cache_FlashAttention_longbench.py   # Llama model implementation with PCW
├── modeling_qwen2_with_pcw_kv_cache_FlashAttention_longbench.py   # Qwen2 model implementation with PCW
├── metrics.py                                              # Evaluation metrics
├── eval_longbench.py                                       # LongBench dataset evaluation
├── eval_infinitebench.py                                   # InfiniteBench dataset evaluation
├── utils.py                                                # General utilities
├── constants.py                                            # Project constants
├── run_test_longbench_multi_gpu_window8_llama.sh          # Llama evaluation script
├── run_test_longbench_multi_gpu_window8_qwen.sh           # Qwen evaluation script
├── scripts/                                                # GPU configuration files
│   ├── gpu_*.yaml                                         # GPU configuration files
│   ├── longbench_metrics.sh                               # LongBench metrics script
│   └── infinitebench_metrics.sh                           # InfiniteBench metrics script
├── longbench_config/                                       # LongBench configurations
│   ├── dataset2*.json                                     # Dataset configuration files
│   ├── model2maxlen*.json                                  # Model configuration files
│   └── past/                                              # Historical configurations
├── datasets/                                               # Dataset storage
│   ├── LongBench/                                         # LongBench dataset
│   └── gsm8k/                                             # GSM8K dataset
├── my_utils/                                               # Utilities
│   ├── logger.py                                          # Logging utilities
│   ├── entropy_utils.py                                   # Entropy calculation utilities
│   ├── cache_revise.py                                    # Cache revision utilities
│   └── priorityqueue.py                                   # Priority queue implementation
├── requirements.txt                                        # Python dependencies
└── requirements.in                                         # Dependency source file
```

## 📊 Results

ParallelComp achieves significant improvements in long-context tasks:

- **91.17%** of GPT-4's performance using 8B model trained on 8K context
- **23.50x** acceleration in prefilling stage
- **1.76x** improvement in chunk throughput
- Outperforms Claude-2 and Kimi-Chat on long-context benchmarks

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{xiong2025parallelcomp,
  title={ParallelComp: Parallel Long-Context Compressor for Length Extrapolation},
  author={Xiong, Jing and Shen, Jianghan and Zheng, Chuanyang and Wan, Zhongwei and Zhao, Chenyang and Yang, Chiwun and Ye, Fanghua and Yang, Hongxia and Kong, Lingpeng and Wong, Ngai},
  journal={arXiv preprint arXiv:2502.14317},
  year={2025}
}
```



## 📞 Contact

For questions and support, please open an issue in this repository or contact the authors.

---

**Note**: This implementation will be fully released soon. Stay tuned for updates!
