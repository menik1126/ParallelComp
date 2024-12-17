import argparse
import logging
from typing import List, Optional

from accelerate import Accelerator
from accelerate.utils import gather_object
import pandas as pd
from transformers import PreTrainedTokenizerBase

from datasets_loader import DATASET_NAMES2LOADERS
from experiment_manager import ExperimentManager
from model_loaders import load_pcw_wrapper
from utils import get_max_n_shots, filter_extremely_long_samples, save_results
   
import torch.distributed as dist
from datetime import timedelta
dist.init_process_group(
        backend='nccl',
        timeout=timedelta(hours=2)
    )
accelerator = Accelerator()
_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_dataset(dataset: str, tokenizer: PreTrainedTokenizerBase, prompt_method: str, sample_method: str, sample_number: int) -> (pd.DataFrame, pd.DataFrame, List):
    print("dataset:{}".format(dataset))
    da = DATASET_NAMES2LOADERS[dataset](sample_method, sample_number)
    print("da:{}".format(da))
    print("da.train_df:{}".format(da.train_df))
    #assert 1==0
    # Filter extremely long samples from both train and test samples:
    _logger.info("filtering test set:")
    test_df = filter_extremely_long_samples(da.test_df, tokenizer, prompt_method=prompt_method)
    _logger.info("filtering train set:")
    train_df = filter_extremely_long_samples(da.train_df, tokenizer, prompt_method=prompt_method)
#    print("da.labels.tolist():{}".format(list(da.labels)))
#    assert 1==0
    print("LEN OF test_df2:{}".format(len(test_df)))
    if prompt_method == "complex_cot" or prompt_method == "complex_cot_pcw" or prompt_method == "complex_cot_pcw_pre_process_window_cache" or prompt_method == "complex_cot_pcw_multi_windows":
        return test_df, train_df, None
    else:
        return test_df, train_df, da.labels


def run_pcw_experiment(dataset: str, model: str, cache_dir: str, subsample_test_set: int, output_dir: str,
                       n_windows: List[int], n_shots_per_window: Optional[int], n_runs: int,
                       random_seed: int, right_indentation: bool, prompt_method: str, output_json: str, model_class: str, sample_method: str, sample_number: int, extra_sample_number: int) -> None:
    print("n_windows:{}".format(n_windows))
    print("prompt_method:{}".format(prompt_method))
#    assert 1==0
    # 在这里传入模型的类
    pcw_model = load_pcw_wrapper(model, cache_dir, right_indentation, max(n_windows), prompt_method=prompt_method, model_class=model_class, accelerator=accelerator)

    test_df, train_df, labels = get_dataset(dataset, pcw_model.tokenizer, prompt_method, sample_method=sample_method, sample_number=sample_number)
    
    if n_shots_per_window is None:
        print("=========pcw_model.context_window_size:{}".format(pcw_model.context_window_size))
        # default behaviour: we take the maximum number of samples per window
        n_shots_per_window = get_max_n_shots(train_df, test_df, pcw_model.tokenizer, pcw_model.context_window_size)
        # 得到每个窗口最多可容纳的样本的数量
        _logger.info(f"Found max n shot per window = {n_shots_per_window}")
    print("n_windows:{}".format(n_windows))   # Number of parallel context windows   不知道为啥是n_windows:[1, 3], 有两种上下文窗口模式？
    print("len(n_windows):{}".format(len(n_windows)))
    print("n_shots_per_window:{}".format(n_shots_per_window))
#    assert 1==0
    n_shots = [i * n_shots_per_window for i in n_windows]
    # labels里面包含分类任务所有的标签
    # n_shots 总共可以装下的上下文的数
    print("n_shots:{}".format(n_shots))
    #max_length = tokenizer.model_max_length
    print("LEN OF test_df3:{}".format(len(test_df)))
    em = ExperimentManager(test_df, train_df, pcw_model, labels, random_seed=random_seed,
                           n_shots_per_window=n_shots_per_window, subsample_test_set=subsample_test_set, prompt_method=prompt_method, output_json=output_json, accelerator=accelerator, n_windows=max(n_windows), sample_method=sample_method, sample_number=sample_number, extra_sample_number=extra_sample_number)

    accuracies = em.run_experiment_across_shots(n_shots, n_runs)
    save_results(dataset, n_shots, accuracies, output_dir, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', action='store', required=True,
                        help=f'Name of dataset (for example sst2).'
                             f' The supported datasets are: {DATASET_NAMES2LOADERS.keys()}')
    parser.add_argument('--model', dest='model', action='store', default='gpt2',
                        help='HF model name to use, either gpt2 or LLaMa family models')
    parser.add_argument('--subsample-test-set', dest='subsample_test_set', action='store', required=False, type=int,
                        help='Size of test set to use to speed up eval. None means using all test set.')
    parser.add_argument('--output-dir', dest='output_dir', required=False, help="Directory for saving the results",
                        default='./temp', action='store', type=str)
    parser.add_argument('--output_json', required=False, help="Directory for saving the results",
                        default='./output.json', action='store', type=str)
    parser.add_argument('--model_class', required=False, help="decide what model class to use",
                        default='modeling_llama_with_pcw', type=str)
    parser.add_argument('--prompt_method', required=False, help="decide what prompt method to use",
                        default='complex_cot', type=str)
    parser.add_argument('--sample_method', required=False, help="decide what prompt method to use",
                        default='sample', type=str)
    parser.add_argument('--sample_number', required=False, help="decide what prompt method to use",
                        default=64, type=int)
    
    parser.add_argument('--extra_sample_number', required=False, help="decide what prompt method to use",
                        default=1, type=int)
    parser.add_argument('--cache-dir', help="Hugging face cache dir", type=str, default=None, dest='cache_dir')
    
    parser.add_argument('--random-seed', dest='random_seed', required=False, default=42, action='store', type=int)
    parser.add_argument('--n-runs', dest='n_runs',
                        help="Number of times experiments are repeated for every number of windows", action='store',
                        type=int, default=1)
    parser.add_argument('-n', '--n-windows', dest='n_windows', help="Number of parallel context windows",
                        action='append', type=int)
    parser.add_argument('--n-shots-per-window', dest='n_shots_per_window',
                        help="number of examples to fit in each window", type=int, default=None)
    parser.add_argument('--right-indentation', dest='right_indentation', help="ident all windows to the right",
                        action='store_true', default=False)
    args = parser.parse_args()
    run_pcw_experiment(**vars(args))
