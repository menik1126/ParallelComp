import argparse
import logging
from typing import List, Optional
import json
from accelerate import Accelerator
from accelerate.utils import gather_object
import pandas as pd
from transformers import PreTrainedTokenizerBase
import re 
from datasets_loader import DATASET_NAMES2LOADERS
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

# if dist.is_initialized():
#     dist.destroy_process_group()
# dist.init_process_group(
#         backend='nccl',
#         timeout=timedelta(hours=2)
# )
# accelerator = Accelerator()
accelerator = Accelerator(kwargs_handlers=[kwargs])


logger = Logger(accelerator)
logger.set_console_level(logging.DEBUG)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 设置 PYTORCH_CUDA_ALLOC_CONF
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# STOP_SEQUENCE = '\n'

# def get_dataset(dataset: str, tokenizer: PreTrainedTokenizerBase, prompt_method: str, sample_method: str, sample_number: int) -> (pd.DataFrame, pd.DataFrame, List):
#     print("dataset:{}".format(dataset))
#     da = DATASET_NAMES2LOADERS[dataset](sample_method, sample_number)
#     print("da:{}".format(da))
#     print("da.train_df:{}".format(da.train_df))
#     #assert 1==0
#     # Filter extremely long samples from both train and test samples:
#     _logger.info("filtering test set:")
#     test_df = filter_extremely_long_samples(da.test_df, tokenizer, prompt_method=prompt_method)
#     _logger.info("filtering train set:")
#     train_df = filter_extremely_long_samples(da.train_df, tokenizer, prompt_method=prompt_method)
# #    print("da.labels.tolist():{}".format(list(da.labels)))
# #    assert 1==0
#     print("LEN OF test_df2:{}".format(len(test_df)))
#     if prompt_method == "complex_cot" or prompt_method == "complex_cot_pcw" or \
#         prompt_method == "complex_cot_pcw_pre_process_window_cache" or prompt_method == "complex_cot_pcw_multi_windows"\
#             or prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
#         return test_df, train_df, None
#     else:
#         return test_df, train_df, da.labels

def run_pcw_experiment(dataset: str, model: str, cache_dir: str, subsample_test_set: int, output_dir: str,
                       n_windows: List[int], n_shots_per_window: Optional[int], n_runs: int,
                       random_seed: int, right_indentation: bool, prompt_method: str, output_json: str, 
                       model_class: str, sample_method: str, sample_number: int, extra_sample_number: int, 
                       data_name: str,
                       # attention calibration
                       input_stitching:bool=False,calibration_mode:int=0,calibration_stage:str=None,
                       # base model parameters
                       Truncation_Method:str="",parallel_pattern:str=None, attn_implementation:str=None,
                       special_token:bool=True,key_no_rope:bool=False,
                       attn_avg:bool=False,in_eager_mode:bool=False,
                       delete_context_prompt:bool=False,draw_pic:bool=False,
                       # kv cache parameters
                       capacity:int=512, raw_position_select: bool =True,kv_cache_eviction: bool=False,
                       kv_cache_dynamic:bool=False,recent_token:int=8,stage_eviction:bool=False,
                       # window parameters
                       window_pe_shift:bool=False,del_val:bool=False,
                       rank_windows:bool=False,topk_windows:int=8,rank_cache:bool=False,
                       dynamic_window:bool=False,window_ascend:bool=False,
                       reduce_length:bool=False,reduce_factor:int=1,
                       query_rank:bool=False,query_recent_tokens:int=0,
                       get_recent_attn:bool=False,recent_top_num:int=0,
                       decomposition_factor:int=1,
                       # position shift parameters
                       position_shift:bool=False,shift_factor:int = 2,
                       interval_shift:int=0,
                       # positional_sorting
                       positional_sorting:str=None,rank_ascend=False,
                       # other try
                       parallel_decoding:bool=False,
                       NTK_aware:bool=False,
                       ) -> None:
    print("n_windows:{}".format(n_windows))
    print("capacity:{}".format(capacity))
    #assert 1==0
    # # load config
    model2prompt = json.load(open("longbench_config/dataset2prompt_raw.json", "r"))
    dataset2maxlen = json.load(open("longbench_config/dataset2maxlen.json", "r"))
    model2maxlen = json.load(open("longbench_config/model2maxlen.json", "r"))
    model2maxlem_parallel = json.load(open("longbench_config/model2maxlen_parallel.json", "r"))
    
    
    model2prompt_question = json.load(open("longbench_config/dataset2prompt_quesiton.json", "r"))
    # model2prompt_anchor = json.load(open("longbench_config/dataset2prompt_raw_anchor.json", "r"))
    model2prompt_context = json.load(open("longbench_config/dataset2prompt_context.json", "r"))
    model_context_prompt = json.load(open("longbench_config/dataset2prompt_context_prompt.json", "r"))
    
    
    questions = model2prompt_question[dataset]
    # all_template_anchor = model2prompt_anchor[dataset]
    context = model2prompt_context[dataset]
    all_template = model2prompt[dataset]
    
    # templates = {"question": questions, "context": context, "all": all_template, "all_anchor": all_template_anchor}
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
    
    if "every_window_query_input_query_anchor_new_label" in parallel_pattern:
        query_long_datasets = ["repobench-p","triviaqa"]
        query_middle_datasets = ["samsum","passage_retrieval_en"]
        raw_model_max_len = reduce_factor
        if data_name in query_long_datasets:
            context_max_len = raw_model_max_len-1500
        elif data_name in query_middle_datasets:
            context_max_len = raw_model_max_len-800
        else:
            context_max_len = raw_model_max_len-200
    
    
    logger.info(f"context_max_len: {context_max_len}")
    logger.info(f"raw_model_max_len: {raw_model_max_len}")
    output_max_len = dataset2maxlen[dataset]
    logger.info(f"output_max_len: {output_max_len}")
    if data_name == "longbench":
        data_file = f"datasets/LongBench/{dataset}.jsonl"
    elif data_name =="infinitebench":
        data_file = f"datasets/InfiniteBench/{dataset}.jsonl"
    logger.info(f"parallel_pattern: {parallel_pattern}")
    
    # load model
    pcw_model = load_pcw_wrapper(model, cache_dir, max(n_windows), 
                                 # base model parameters
                                 prompt_method=prompt_method, model_class=model_class, 
                                 accelerator=accelerator, Truncation_Method=Truncation_Method,
                                 parallel_pattern=parallel_pattern,attn_implementation=attn_implementation,
                                 raw_model_max_len=raw_model_max_len,special_token=special_token,
                                 key_no_rope=key_no_rope,draw_pic=draw_pic,
                                 attn_avg=attn_avg,in_eager_mode=in_eager_mode,
                                 delete_context_prompt=delete_context_prompt,context_prompt=model_context_prompt[dataset],
                                 # attention calibration
                                 input_stitching=input_stitching,calibration_mode=calibration_mode,
                                 calibration_stage=calibration_stage,
                                 # kv cache parameters
                                 kv_cache_eviction=kv_cache_eviction,kv_cache_dynamic=kv_cache_dynamic,
                                 stage_eviction=stage_eviction,capacity=capacity,
                                 # window parameters
                                 del_val=del_val,rank_windows=rank_windows,rank_cache=rank_cache,
                                 dynamic_window=dynamic_window,topk_windows=topk_windows,decomposition_factor=decomposition_factor,
                                 window_ascend=window_ascend,
                                 query_rank=query_rank,query_recent_tokens=query_recent_tokens,
                                 get_recent_attn=get_recent_attn,recent_top_num=recent_top_num,
                                 # position shift parameters
                                 position_shift=position_shift,shift_factor=shift_factor,
                                 interval_shift=interval_shift,
                                 # positional_sorting
                                 positional_sorting=positional_sorting,rank_ascend=rank_ascend,
                                 # other try
                                 window_pe_shift=window_pe_shift,
                                 right_indentation=right_indentation,
                                 parallel_decoding=parallel_decoding,
                                 NTK_aware=NTK_aware,
                                 )
    
    # windows_splits
    em = ExperimentManager_longbench(data_file, pcw_model,random_seed=random_seed,
                                     n_windows=max(n_windows),
                                     # base model parameters
                                     context_max_len=context_max_len,raw_model_max_len=raw_model_max_len,
                                     prompt_method=prompt_method,model_class=model_class,
                                     model_name = model,dataset=dataset,parallel_pattern=parallel_pattern,
                                     accelerator=accelerator,model2prompt=model2prompt,templates=templates, 
                                     Truncation_Method=Truncation_Method,attn_implementation=attn_implementation,
                                     special_token=special_token,data_name=data_name,
                                     delete_context_prompt=delete_context_prompt,
                                     attn_avg=attn_avg,
                                     # attention calibration
                                     calibration_mode=calibration_mode,
                                     calibration_stage=calibration_stage,
                                     # kv cache parameters
                                     raw_position_select=raw_position_select,
                                     capacity=capacity,recent_token=recent_token,
                                     kv_cache_eviction=kv_cache_eviction,kv_cache_dynamic=kv_cache_dynamic,
                                     stage_eviction=stage_eviction,
                                     # window parameters
                                     rank_windows=rank_windows,rank_cache=rank_cache,topk_windows=topk_windows,
                                     dynamic_window=dynamic_window,window_ascend=window_ascend,decomposition_factor=decomposition_factor,
                                     
                                     reduce_length=reduce_length,reduce_factor=reduce_factor,
                                     get_recent_attn=get_recent_attn,recent_top_num=recent_top_num,
                                     query_rank=query_rank,query_recent_tokens=query_recent_tokens,
                                     # position shift parameters
                                     position_shift=position_shift,shift_factor=shift_factor,
                                     interval_shift=interval_shift,
                                     # positional_sorting
                                     key_no_rope=key_no_rope,
                                     positional_sorting=positional_sorting,rank_ascend=rank_ascend,
                                     # NTK
                                     NTK_aware=NTK_aware,
                                     )
    em.run_experiment(batch_size=1,output_max_len=output_max_len)
# import torch.distributed as dist
if __name__ == '__main__':

    # dist.init_process_group(
    #      backend='nccl',
    #      timeout=timedelta(hours=2)
    # )
    if dist.is_initialized():
        print(f"Distributed group initialized with timeout: {dist.get_backend()}")
        default_group = dist.distributed_c10d._get_default_group()  # 获取默认分布式进程组
        timeout = default_group.options._timeout                    # 获取超时时间
        print(f"Distributed group timeout: {timeout}")
    # assert 1==0
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
    parser.add_argument('--data_name', required=False, help="data_name",
                        default='longbench', action='store', type=str)
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
    parser.add_argument('--positional_sorting', required=False, help="decide what positional_sorting method to use",
                        default=None,type=str)
    parser.add_argument('--raw_position_select', type=lambda x: x.lower() == 'true', 
                    default=False, help="Enable or disable the specific feature (True/False).")
    parser.add_argument('--kv_cache_eviction', type=lambda x: x.lower() == 'true', 
                    default=False, help="Decide whether to adopt the KV cache eviction method.")
    parser.add_argument('--kv_cache_dynamic', type=lambda x: x.lower() == 'true', 
                    default=False, help="Decide whether to adopt the KV cache eviction method.")
    parser.add_argument('--recent_token', required=False, help="decide what prompt method to use",
                        default=8, type=int)
    parser.add_argument('--parallel_decoding', required=False, help="decide using parallel decoding window",
                        default=False, type=bool)
    parser.add_argument('--window_pe_shift', required=False, help="decide using parallel decoding window",
                        default=False, type=bool)
    parser.add_argument('--attn_implementation', 
                    required=False, 
                    help="Decide using parallel decoding window", 
                    choices=['eager', 'flash_attention_2'],  
                    default='flash_attention_2')  

    parser.add_argument('--del_val', type=lambda x: x.lower() == 'true', 
                    default=False, help="Decide whether to adopt the KV cache eviction method.")
    parser.add_argument('--rank_windows', type=lambda x: x.lower() == 'true', 
                    default=False, help="Decide whether to rank the  windows.")
    parser.add_argument('--topk_windows', required=False, help="get topk_windows ",
                        default=8, type=int)
    parser.add_argument('--rank_cache', type=lambda x: x.lower() == 'true', 
                    default=False, help="rank_cache.")
    parser.add_argument('--stage_eviction', type=lambda x: x.lower() == 'true', 
                    default=False, help="stage_eviction.")
    parser.add_argument('--dynamic_window', type=lambda x: x.lower() == 'true', 
                    default=False, help="dynamic_window.")
    parser.add_argument('--window_ascend', type=lambda x: x.lower() == 'true', 
                    default=False, help="window_ascend.")
    parser.add_argument('--NTK_aware', type=lambda x: x.lower() == 'true', 
                    default=False, help="NTK_aware.")
    parser.add_argument('--position_shift', type=lambda x: x.lower() == 'true', 
                    default=False, help="position_shift.")
    parser.add_argument('--shift_factor', required=False, help="decide what prompt method to use",
                        default=0, type=int)
    parser.add_argument('--reduce_length', type=lambda x: x.lower() == 'true', 
                    default=False, help="reduce_length.")
    parser.add_argument('--reduce_factor', required=False, help="decide what prompt method to use",
                        default=1, type=int)
    parser.add_argument('--query_rank', type=lambda x: x.lower() == 'true', 
                    default=False, help="query_rank.")
    parser.add_argument('--special_token', type=lambda x: x.lower() == 'true', 
                    default=True, help="special_token.")
    parser.add_argument('--get_recent_attn', type=lambda x: x.lower() == 'true',
                    default=False, help="get_recent_attn.")
    parser.add_argument('--key_no_rope', type=lambda x: x.lower() == 'true',
                    default=False, help="key_no_rope.")
    parser.add_argument('--rank_ascend', type=lambda x: x.lower() == 'true',
                    default=False, help="rank_ascend.")
    parser.add_argument('--delete_context_prompt', type=lambda x: x.lower() == 'true',
                    default=False, help="delete_context_prompt.")
    parser.add_argument('--attn_avg', type=lambda x: x.lower() == 'true',
                    default=False, help="attn_avg.")
    parser.add_argument('--in_eager_mode', type=lambda x: x.lower() == 'true',
                    default=False, help="in_eager_mode.")
    parser.add_argument('--input_stitching', type=lambda x: x.lower() == 'true',
                    default=False, help="input_stitching.")
    parser.add_argument('--draw_pic', required=False, help="decide what prompt method to use",
                    default=0, type=int)
    parser.add_argument('--recent_top_num', required=False, help="decide what prompt method to use",
                    default=0, type=int)
    parser.add_argument('--interval_shift', required=False, help="decide what prompt method to use",
                    default=0, type=int)
    parser.add_argument('--decomposition_factor', required=False, help="decide what prompt method to use",
                    default=1, type=int)
    parser.add_argument('--query_recent_tokens', required=False, help="decide what prompt method to use",
                    default=0, type=int)
    parser.add_argument('--calibration_mode', required=False, help="decide what prompt method to use",
                    default=0, type=int)
    parser.add_argument('--calibration_stage', required=False, help="decide what prompt method to use",
                    default=None, type=str)
    args = parser.parse_args()
    # 参数检查
    if args.parallel_pattern != "default":
        # stage_eviction 与 kv_cache_eviction互斥（第二阶段驱逐）
        if args.stage_eviction == True and args.kv_cache_eviction == True:
            raise ValueError("stage_eviction and kv_cache_eviction are mutually exclusive")
        # 动态驱逐前先要有prefill驱逐
        if args.kv_cache_dynamic == True and not(args.stage_eviction == True or args.kv_cache_eviction == True):
            raise ValueError("kv_cache_dynamic must be used with kv_cache_eviction or stage_eviction")
        # if args.stage_eviction == True:
        #     if args.parallel_pattern!="every_window_no_query_input_query_stage":
        #         raise ValueError("stage_eviction must be used with every_window_no_query_input_query_stage")
        if args.kv_cache_eviction == True:
            if args.parallel_pattern=="every_window_no_query_input_query_stage":
                raise ValueError("kv_cache_eviction wrong")
    # 窗口动态驱逐，rank_windows需要为true
    if args.dynamic_window == True:
        if not args.rank_windows: # 需要启用rank_windows topk_windows有效
            raise ValueError("rank_windows must be true when dynamic_window is true")
        # if args.topk_windows <= 0:
        #     raise ValueError("topk_windows must be greater than 0")
    if args.window_ascend==True:
        if not args.rank_cache: # 启动排序缓存
            raise ValueError("rank_cache must be true when window_ascend is true")
    # 如果使用query进行定级，需要rank_cache
    if args.query_rank:
        if "anchor_new" not in args.parallel_pattern:
            raise ValueError("query_rank must be used with anchor_new")
    
    if "default" in args.parallel_pattern:
        args.Truncation_Method = None
        args.dynamic_window = False
        args.rank_windows = False
        args.rank_cache = False
        args.stage_eviction = False
    else:
        args.Truncation_Method = "WO_Truncation"
    # 并行解码
    if "parallel" in args.parallel_pattern:
        args.parallel_decoding = True
    if not args.get_recent_attn: # 不获取最近的attention,recent_top_num必须为0
        args.recent_top_num = 0
    # 位置编码的排序
    if args.positional_sorting is not None and args.positional_sorting.lower() !="none":
        args.key_no_rope = True
        if args.positional_sorting.lower() =="ntk":
            args.NTK_aware = True
    if "default" in args.parallel_pattern:
        if "NTK" in args.parallel_pattern:
            args.NTK_aware = True
    
    if args.query_rank and args.get_recent_attn:
        raise ValueError("query_rank and get_recent_attn are mutually exclusive")
    if args.calibration_mode == 2:
        args.in_eager_mode = True
    
    if args.dataset == "passage_retrieval_en":
        args.topk_windows = -1
    
    if args.positional_sorting == "chunkllama":
        args.model_class = "modeling_llama_with_pcw_kv_cache_FlashAttention_longbench_437"
    # 这些参数必须符合条件(弃用参数)
    if not (args.rank_windows and not args.rank_cache and not args.window_ascend ) and "default" not in args.parallel_pattern:
        # 启动rank筛选,不启动排序缓存，不启动窗口升序
        if args.decomposition_factor!=1:
            raise ValueError("decomposition_factor must be 1")
        raise ValueError("rank_windows must be true, rank_cache and window_ascend must be false")
    # if not args.special_token:
    #     raise ValueError("special_token must be true")
    # if args.reduce_length:
    #     raise ValueError("reduce_length must be false")
    # if args.get_recent_attn:
    #     raise ValueError("get_recent_attn must be false")
    # if args.recent_top_num!=0:
    #     raise ValueError("recent_top_num must be 0")
    
    logger.info(f"key_no_rope:{args.key_no_rope}")
    run_pcw_experiment(**vars(args))
