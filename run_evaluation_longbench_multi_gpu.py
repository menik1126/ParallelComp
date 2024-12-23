import argparse
import logging
from typing import List, Optional
import json
from accelerate import Accelerator
from accelerate.utils import gather_object
import pandas as pd
from transformers import PreTrainedTokenizerBase

from datasets_loader import DATASET_NAMES2LOADERS
from experiment_manager import ExperimentManager,ExperimentManager_longbench
from model_loaders import load_pcw_wrapper
from utils import get_max_n_shots, filter_extremely_long_samples, save_results
from my_utils.logger import Logger
import torch.distributed as dist
from datetime import timedelta
# dist.init_process_group(
#         backend='nccl',
#         timeout=timedelta(hours=2)
#     )
accelerator = Accelerator()
logger = Logger()
logger.set_console_level(logging.DEBUG)

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
    if prompt_method == "complex_cot" or prompt_method == "complex_cot_pcw" or \
        prompt_method == "complex_cot_pcw_pre_process_window_cache" or prompt_method == "complex_cot_pcw_multi_windows"\
            or prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
        return test_df, train_df, None
    else:
        return test_df, train_df, da.labels

def run_pcw_experiment(dataset: str, model: str, cache_dir: str, subsample_test_set: int, output_dir: str,
                       n_windows: List[int], n_shots_per_window: Optional[int], n_runs: int,
                       random_seed: int, right_indentation: bool, prompt_method: str, output_json: str, 
                       model_class: str, sample_method: str, sample_number: int, extra_sample_number: int, 
                       capacity:int, parallel_pattern:str, Truncation_Method:str) -> None:
    print("n_windows:{}".format(n_windows))
    # load model
    pcw_model = load_pcw_wrapper(model, cache_dir, right_indentation, max(n_windows), prompt_method=prompt_method, model_class=model_class, accelerator=accelerator, capacity=capacity, Truncation_Method=Truncation_Method)
    
    # # load config
    model2prompt = json.load(open("longbench_config/dataset2prompt_raw.json", "r"))
    dataset2maxlen = json.load(open("longbench_config/dataset2maxlen.json", "r"))
    model2maxlen = json.load(open("longbench_config/model2maxlen.json", "r"))
    
    model2prompt_question = json.load(open("longbench_config/dataset2prompt_quesiton.json", "r"))
    model2prompt_context = json.load(open("longbench_config/dataset2prompt_context.json", "r"))
    
    questions = model2prompt_question[dataset]
    context = model2prompt_context[dataset]
    all_template = model2prompt[dataset]
    
    templates = {"question": questions, "context": context, "all": all_template }
    logger.info("loading datasets finished")
    
    # load dataset
    model_path = model.lower()
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
    logger.info(f"model_max_len: {model_max_len}")
    
    output_max_len = dataset2maxlen[dataset]
    logger.info(f"output_max_len: {output_max_len}")
    data_file = f"datasets/LongBench/{dataset}.jsonl"
    logger.info(f"parallel_pattern: {parallel_pattern}")
    
    # windows_splits
    em = ExperimentManager_longbench(data_file, pcw_model,random_seed=random_seed,
                                     n_windows=max(n_windows),
                                     max_token_length=model_max_len,
                                     prompt_method=prompt_method,model_class=model_class,
                                     model_name = model,dataset=dataset,parallel_pattern=parallel_pattern,
                                     accelerator=accelerator,
                                     model2prompt=model2prompt,templates=templates, Truncation_Method=Truncation_Method
                                     )
    em.run_experiment(batch_size=1,output_max_len=output_max_len)

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
    
    parser.add_argument('--capacity', required=False, help="decide what prompt method to use",
                        default=512, type=int)
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
    parser.add_argument('--parallel_pattern', required=False, help="decide what prompt method to use",
                        default='',type=str)
    parser.add_argument('--Truncation_Method', required=False, help="decide what Truncation_Method method to use",
                        default='',type=str)
                        
    args = parser.parse_args()
    run_pcw_experiment(**vars(args))
