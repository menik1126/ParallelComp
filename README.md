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

- Python 3.8+
- PyTorch 2.0.1
- CUDA compatible GPU(s)
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
python run_evaluation.py \
    --dataset sst2 \
    --model microsoft/DialoGPT-medium \
    --n-windows 1 \
    --subsample-test-set 100 \
    --output-dir ./results \
    --prompt_method complex_cot
```

#### Multi-GPU Evaluation

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --main_process_port 5326 --num_processes 4 --config_file scripts/gpu_4.yaml \
    run_evaluation_longbench_multi_gpu.py \
    --dataset narrativeqa \
    --n-windows 2 \
    --subsample-test-set 2000 \
    --model /path/to/llama2chat \
    --prompt_method complex_cot_pcw_multi_windows \
    --model_class modeling_llama_with_pcw
```

### Supported Models

- **GPT-2** family models
- **LLaMA** family models  
- **Gemma** models
- **Qwen2** models

### Supported Datasets

#### Long Context Benchmarks
- **LongBench**: narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p
- **Standard Benchmarks**: SST-2, GSM8K



## 📈 Evaluation Scripts

### LongBench Evaluation

Evaluate on LongBench datasets using the provided scripts:

```bash
# Multi-GPU evaluation 
bash run_test_longbench_multi_gpu_window8.sh

# Single GPU evaluation  
bash run_test_longbench_single_gpu_window8.sh
```

For single GPU:

```bash
bash run_test_longbench_single_gpu_window8.sh
```

### Evaluation Metrics

Calculate metrics for evaluation results:

```bash
# For LongBench results
bash scripts/longbench_metrics.sh --results_dir ./results --new_method your_method_name --switch true

# For InfiniteBench results  
bash scripts/infinitebench_metrics.sh --results_dir ./results --new_method your_method_name --switch true

# For GSM8K or other tasks
bash scripts/evaluate_per_result.sh
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

### Custom Evaluation

```python
from model_loaders import load_pcw_wrapper
from experiment_manager import ExperimentManager

# Load model
model = load_pcw_wrapper(
    model_name="microsoft/DialoGPT-medium",
    cache_dir="./cache",
    right_indentation=False,
    n_windows=2,
    prompt_method="complex_cot_pcw"
)

# Run experiments
em = ExperimentManager(test_df, train_df, model, labels)
accuracies = em.run_experiment_across_shots(n_shots=[4, 8], n_runs=3)
```

## 🔧 Advanced Configuration

### Attention Calibration

ParallelComp includes attention calibration strategies to mitigate attention sink issues:

```python
# Configure attention calibration in your model
model = load_pcw_wrapper(
    model_name="your_model",
    calibration_strategy="attention_bias_reduction", 
    chunk_eviction=True
)
```

### Memory Management

For ultra-long contexts, enable chunk eviction:

```bash
python run_evaluation.py \
    --enable-chunk-eviction \
    --max-memory-usage 0.8 \
    --parallel-kv-cache
```

## 🚧 TODO & Roadmap

- [ ] **Code Organization**: Currently organizing and cleaning up the codebase for better usability
- [ ] **Gemma Support**: Adding full support for Gemma model family
- [ ] **SGLang Integration**: Adding support for SGLang inference engine for improved performance
- [ ] **Documentation**: Expanding documentation with more detailed examples
- [ ] **Benchmarks**: Adding more comprehensive benchmark results

## 📁 Project Structure

```
ParallelComp/
├── modeling_*.py              # Model implementations with PCW
├── run_evaluation*.py         # Evaluation scripts
├── experiment_manager*.py     # Experiment management
├── pcw_wrapper*.py           # Parallel Context Window wrapper
├── datasets_loader.py        # Dataset loading utilities
├── eval_longbench.py         # LongBench evaluation
├── metrics.py                # Evaluation metrics
├── scripts/                  # Bash scripts and configs
├── longbench_config/         # LongBench configurations
├── results/                  # Output results
└── requirements.txt          # Dependencies
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
