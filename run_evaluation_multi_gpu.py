import argparse
import logging
from typing import List, Optional
import json
from functools import  partialmethod
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
import pandas as pd
from transformers import PreTrainedTokenizerBase
import re 
# from datasets_loader import DATASET_NAMES2LOADERS
from experiment_manager import ExperimentManager_longbench
from model_loaders import load_pcw_wrapper
from utils import get_max_n_shots, filter_extremely_long_samples, save_results
from my_utils.logger import Logger
import torch.distributed as dist
from datetime import timedelta
import os
import pandas as pd
# dist.init_process_group(
#         backend='nccl',
#         timeout=timedelta(hours=2)
# )
from accelerate.utils import InitProcessGroupKwargs
kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2),backend="nccl")
from my_utils.cache_revise import from_legacy_cache, get_seq_length
import transformers
# if dist.is_initialized():
#     dist.destroy_process_group()
# dist.init_process_group(
#         backend='nccl',
#         timeout=timedelta(hours=2)
# )
# accelerator = Accelerator()
accelerator = Accelerator(kwargs_handlers=[kwargs])

NUM_HEAD_NUMS = {
    "meta-llama/Llama-2-7b-chat-hf": 32,
    "~/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct": 32,
    "Qwen/Qwen2.5-7B-Instruct": 28,
    "Qwen/Qwen3-8B": 32,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
}
NUM_LAYER_NUMS = {
    "meta-llama/Llama-2-7b-chat-hf": 32,
    "~/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct": 32,
    "Qwen/Qwen2.5-7B-Instruct": 28,
    "Qwen/Qwen3-8B": 36,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
}


logger = Logger(accelerator)
logger.set_console_level(logging.DEBUG)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 设置 PYTORCH_CUDA_ALLOC_CONF
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# STOP_SEQUENCE = '\n'

def run_pcw_experiment(dataset: str, model: str, cache_dir: str,
                       n_windows: List[int], random_seed: int, 
                       model_class: str,data_name: str,reduce_factor:int=1,
                       # attention calibration
                       calibration_mode:int=0,calibration_stage:str=None,
                       # base model parameters
                       parallel_pattern:str=None,special_token:bool=True,
                       # kv cache parameters
                       capacity:int=512,kv_cache_eviction: bool=False,
                       kv_cache_dynamic:bool=False,recent_token:int=8,stage_eviction:bool=False,
                       # window parameters
                       topk_windows:int=8,
                       query_rank:bool=False,query_recent_tokens:int=0,
                       ) -> None:
    print("n_windows:{}".format(n_windows))
    print("capacity:{}".format(capacity))
    model2prompt = json.load(open("longbench_config/dataset2prompt_raw.json", "r"))
    dataset2maxlen = json.load(open("longbench_config/dataset2maxlen.json", "r"))
    model2maxlen = json.load(open("longbench_config/model2maxlen.json", "r"))
    model2maxlem_parallel = json.load(open("longbench_config/model2maxlen_parallel.json", "r"))
    
    model2prompt_question = json.load(open("longbench_config/dataset2prompt_quesiton.json", "r"))
    model2prompt_context = json.load(open("longbench_config/dataset2prompt_context.json", "r"))
    model_context_prompt = json.load(open("longbench_config/dataset2prompt_context_prompt.json", "r"))
    
    questions = model2prompt_question[dataset]
    context = model2prompt_context[dataset]
    all_template = model2prompt[dataset]
    
    templates = {"question": questions, "context": context, "all": all_template}
    logger.info("loading datasets finished")
    
    # load dataset
    model_path = model.lower()
    for key in model2maxlen:
        if key in model_path:
            context_max_len = model2maxlem_parallel[key]
    for key in model2maxlem_parallel:
        if key in model_path:
            raw_model_max_len = model2maxlen[key]
    
    if "ppl" in parallel_pattern and "default" in parallel_pattern:
        raw_model_max_len = reduce_factor
    
    if "default_label" in parallel_pattern:
        raw_model_max_len = reduce_factor
    query_long_datasets = ["repobench-p","triviaqa"]
    query_middle_datasets = ["samsum","passage_retrieval_en"]
    if dataset in query_long_datasets and "uncomp" in calibration_stage:
        calibration_stage += "_long"
    
    if "parallel_comp_label" in parallel_pattern:
        raw_model_max_len = reduce_factor
        if dataset in query_long_datasets:
            context_max_len = raw_model_max_len-1500
        elif dataset in query_middle_datasets:
            context_max_len = raw_model_max_len-800
        else:
            context_max_len = raw_model_max_len-200
    
    if dataset in ["passkey","kv_retrieval","number_string"]:
        topk_windows = -1
        
    logger.info(f"context_max_len: {context_max_len}")
    logger.info(f"raw_model_max_len: {raw_model_max_len}")
    output_max_len = dataset2maxlen[dataset]
    logger.info(f"output_max_len: {output_max_len}")
    if data_name == "longbench":
        data_file = f"./datasets/LongBench/{dataset}.jsonl"
    elif data_name =="infinitebench":
        data_file = f"./datasets/InfiniteBench/{dataset}.jsonl"
    logger.info(f"parallel_pattern: {parallel_pattern}")
    # 加载head信息
    if "uncomp" in calibration_stage:
        data_all_layers = []
        data_all_layers_2 = []
        logger.info(f"args.model_path is {model}")
        num_hidden_layers =  NUM_LAYER_NUMS[model]
        num_attention_heads = NUM_HEAD_NUMS[model]
        print("num_attention_heads:{}".format(num_hidden_layers))
        for i in range(num_hidden_layers):
            if "llama-2" in model.lower():            
                filename = "~/UNComp/search/512/llama2-chat/query/head_type_search_layer" + str(i) + ".csv"
            elif "llama-3.1" in model.lower():
                filename = "~/UNComp/search/llama31/svd32/head_type_search_layer" + str(i) + ".csv"
            elif "llama-3" in model.lower():
                filename = "~/UNComp/search/llama3-instruct/2_groups/svd32/head_type_search_layer" + str(i) + ".csv"
            elif "qwen2" in model.lower():
                filename = "~/UNComp/search/qwen2/svd32/head_type_search_layer" + str(i) + ".csv"
            data_layers = []
            if os.path.isfile(filename):
                import csv
                with open(filename, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        data_layers.append([int(value) for value in row])
            else:
                logger.error("load error")
                raise ValueError
            
            data_layers = np.array(data_layers)
            data_layers = data_layers.sum(axis=0)
            num_heads = num_attention_heads // 2
            top_half_indices = np.argpartition(data_layers, -num_heads)[-num_heads:]
            down_half_indices = np.argpartition(data_layers, -num_heads)[:num_heads]
            indices = torch.cat([torch.tensor(down_half_indices).sort()[0],torch.tensor(top_half_indices).sort()[0]])
            if "reverse" in calibration_stage:
                indices = torch.cat([torch.tensor(top_half_indices).sort()[0],torch.tensor(down_half_indices).sort()[0]])
            data_all_layers.append([top_half_indices,down_half_indices])
            data_all_layers_2.append(indices)
        head_datas = torch.from_numpy(np.array(data_all_layers_2)).tolist()
    else:
        head_datas = None
    pcw_model = load_pcw_wrapper(model, cache_dir, max(n_windows), 
                                 # base model parameters
                                 model_class=model_class,accelerator=accelerator, 
                                 parallel_pattern=parallel_pattern,
                                 raw_model_max_len=raw_model_max_len,special_token=special_token,
                                 context_prompt=model_context_prompt[dataset],
                                 # attention calibration
                                 calibration_mode=calibration_mode,
                                 calibration_stage=calibration_stage,
                                 # kv cache parameters
                                 kv_cache_eviction=kv_cache_eviction,kv_cache_dynamic=kv_cache_dynamic,
                                 stage_eviction=stage_eviction,capacity=capacity,
                                 # window parameters
                                 topk_windows=topk_windows,
                                 query_rank=query_rank,query_recent_tokens=query_recent_tokens,
                                 # other try
                                 head_datas=head_datas,
                                 )
    
    em = ExperimentManager_longbench(data_file, pcw_model,random_seed=random_seed,
                                     n_windows=max(n_windows),
                                     # base model parameters
                                     context_max_len=context_max_len,raw_model_max_len=raw_model_max_len,
                                     model_class=model_class,
                                     model_name = model,dataset=dataset,parallel_pattern=parallel_pattern,
                                     accelerator=accelerator,model2prompt=model2prompt,templates=templates, 
                                     special_token=special_token,data_name=data_name,
                                     # attention calibration
                                     calibration_mode=calibration_mode,calibration_stage=calibration_stage,
                                     # kv cache parameters
                                     capacity=capacity,recent_token=recent_token,
                                     kv_cache_eviction=kv_cache_eviction,kv_cache_dynamic=kv_cache_dynamic,
                                     stage_eviction=stage_eviction,
                                     # window parameters
                                     topk_windows=topk_windows,
                                     query_rank=query_rank,query_recent_tokens=query_recent_tokens,
                                     )
    em.run_experiment(batch_size=1,output_max_len=output_max_len)

if __name__ == '__main__':
    if dist.is_initialized():
        print(f"Distributed group initialized with timeout: {dist.get_backend()}")
        default_group = dist.distributed_c10d._get_default_group()  # 获取默认分布式进程组
        timeout = default_group.options._timeout                    # 获取超时时间
        print(f"Distributed group timeout: {timeout}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', action='store', required=True,
                        help=f'Name of dataset (for example sst2).')
    parser.add_argument('--model', dest='model', action='store', default='gpt2',
                        help='HF model name to use, either gpt2 or LLaMa family models')
    parser.add_argument('--data_name', required=False, help="data_name",
                        default='longbench', action='store', type=str)
    parser.add_argument('--model_class', required=False, help="decide what model class to use",
                        default='modeling_llama_with_pcw', type=str)
    parser.add_argument('--capacity', required=False, help="decide what prompt method to use",
                        default=512, type=int)
    parser.add_argument('--cache-dir', help="Hugging face cache dir", type=str, default=None, dest='cache_dir')
    
    parser.add_argument('--random-seed', dest='random_seed', required=False, default=42, action='store', type=int)
    parser.add_argument('-n', '--n-windows', dest='n_windows', help="Number of parallel context windows",
                        action='append', type=int)
    parser.add_argument('--parallel_pattern', required=False, help="decide what prompt method to use",
                        default='',type=str)
    parser.add_argument('--kv_cache_eviction', type=lambda x: x.lower() == 'true', 
                    default=False, help="Decide whether to adopt the KV cache eviction method.")
    parser.add_argument('--kv_cache_dynamic', type=lambda x: x.lower() == 'true', 
                    default=False, help="Decide whether to adopt the KV cache eviction method.")
    parser.add_argument('--recent_token', required=False, help="decide what prompt method to use",
                        default=8, type=int)
    parser.add_argument('--topk_windows', required=False, help="get topk_windows ",
                        default=8, type=int)
    parser.add_argument('--stage_eviction', type=lambda x: x.lower() == 'true', 
                    default=False, help="stage_eviction.")
    parser.add_argument('--reduce_factor', required=False, help="decide what prompt method to use",
                        default=1, type=int)
    parser.add_argument('--query_rank', type=lambda x: x.lower() == 'true', 
                    default=False, help="query_rank.")
    parser.add_argument('--special_token', type=lambda x: x.lower() == 'true', 
                    default=True, help="special_token.")
    parser.add_argument('--query_recent_tokens', required=False, help="decide what prompt method to use",
                    default=0, type=int)
    parser.add_argument('--calibration_mode', required=False, help="decide what prompt method to use",
                    default=0, type=int)
    parser.add_argument('--calibration_stage', required=False, help="decide what prompt method to use",
                    default=None, type=str)
    args = parser.parse_args()
    if args.parallel_pattern != "default":
        if args.stage_eviction == True and args.kv_cache_eviction == True:
            raise ValueError("stage_eviction and kv_cache_eviction are mutually exclusive")
        if args.kv_cache_dynamic == True and not(args.stage_eviction == True or args.kv_cache_eviction == True):
            raise ValueError("kv_cache_dynamic must be used with kv_cache_eviction or stage_eviction")
        if args.kv_cache_eviction == True:
            if args.parallel_pattern=="every_window_no_query_input_query_stage":
                raise ValueError("kv_cache_eviction wrong")
    if args.query_rank:
        if "parallel_comp" not in args.parallel_pattern:
            raise ValueError("query_rank must be used with parallel_comp")
    if "default" in args.parallel_pattern:
        args.stage_eviction = False
    if args.dataset == "passage_retrieval_en":
        args.topk_windows = -1
    
    if "uncomp" in args.calibration_stage:
        transformers.cache_utils.DynamicCache.from_legacy_cache = partialmethod(
            from_legacy_cache, head_group_num=2
        )
        transformers.cache_utils.DynamicCache.get_seq_length = get_seq_length    
    
    run_pcw_experiment(**vars(args))
