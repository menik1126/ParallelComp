from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import random
import torch
from transformers import models
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from logits_processor import RestrictiveTokensLogitsProcessor
from utils import n_tokens_in_prompt
from tqdm import tqdm 
import logging
from my_utils.logger import Logger
from my_utils import priorityqueue
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import HybridCache
import torch.multiprocessing as mp


import torch.nn.functional as F
logger = Logger()
logger.set_console_level(logging.DEBUG)

def combine_past_key_values_longbench(model:PCWModelWrapper,
                                      past_lst: List[Tuple[Tuple[torch.Tensor]]], 
                                      window_eviction: List[Tuple[Tuple[torch.Tensor]]],
                                      query_len, token_prompt_size: int, del_val: bool) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor]]:
#    assert 1==0
    # We eliminate all but one bos token from windows to avoid multiple bos, which deterred our results.
    if not isinstance(past_lst[0], HybridCache):
        n_layers = len(past_lst[0])
        first_window = past_lst[0]
        if first_window == 0:
            query_len = [-first_window[0][0].shape[2]]*len(query_len)
        first_query_len = query_len[0]
        other_windows = past_lst[1:]
        # print("ok")
        with torch.no_grad():
            if del_val != True:
                if model.decomposition_factor != 1:
                    boundary_ss = model.boundary_ss
                    if model.rank_windows and model.rank_cache: 
                        if not model.window_ascend: # 损失值最大的在最前面
                            if model.topk_windows < 0: 
                                temp = boundary_ss.get_elements_ascending()
                            else:
                                temp = boundary_ss.get_elements_descending()
                        else: # 值最小的在最前面
                            if model.topk_windows < 0:
                                temp = boundary_ss.get_elements_descending()
                            else:
                                temp = boundary_ss.get_elements_ascending()
                    elif model.rank_windows and not model.rank_cache: # 默认顺序
                        temp = boundary_ss.get_elements_key("location")
                        if "random" in model.parallel_pattern:
                            temp = boundary_ss.get_elements_random()
                    # print("temp:{}".format(temp))
                    
                    combine_past_key_values = tuple(
                    (
                        torch.cat([ past_lst[item['location']][i][0][:,:,item['boundary'][0]:item['boundary'][1],:] for item in temp ], dim=2),
                        torch.cat([ past_lst[item['location']][i][1][:,:,item['boundary'][0]:item['boundary'][1],:] for item in temp ], dim=2)                    
                    )    
                    for i in range(n_layers)
                    )
                    del boundary_ss
                    # assert 1==0
                else:
                    # print("ok")
                    # print(n_layers)
                    # assert 1==0
                    combine_past_key_values = tuple(
                        (
                        torch.cat([first_window[i][0][:,:,:-first_query_len,:]] + [c[i][0][:, :, 1+token_prompt_size:-query_len[index+1], :]  for index,c in enumerate(other_windows) ], dim=2),
                        torch.cat([first_window[i][1][:,:,:-first_query_len,:]] + [c[i][1][:, :, 1+token_prompt_size:-query_len[index+1], :]  for index,c in enumerate(other_windows)], dim=2)
                        )
                        for i in range(n_layers)
                    )   
            else:
                combine_past_key_values = []
                #assert 1==0
                for i in range(n_layers):
                    # 初始化当前层的 key 和 value
                    # 添加 first_window 的 key 和 value
                    final_key = first_window[i][0][:, :, :-first_query_len, :]#.contiguous()
                    final_value = first_window[i][1][:, :, :-first_query_len, :]#.contiguous()
                    # logger.info(f"first_window[i][0].shape:{first_window[i][0].shape}")
                    # logger.info(f"first_window[i][1].shape:{first_window[i][1].shape}")
                    del list(first_window[i])[0]
                    del list(first_window[i])[1]
                    torch.cuda.empty_cache()
                    # 遍历 other_windows，分段拼接并释放不必要的变量
                    for index, c in enumerate(other_windows):
                        #logger.info(f"i:{i} c:{index}")
                        # if i == 1:
                        #    assert 1==0
                        # logger.info(f"c[i][0].shape:{c[i][0].shape}")
                        # logger.info(f"c[i][1].shape:{c[i][1].shape}")
                        # 拼接当前 window 的 key 和 value
                        current_key = c[i][0][:, :, 1:-query_len[index + 1], :]#.contiguous()
                        current_value = c[i][1][:, :, 1:-query_len[index + 1], :]#.contiguous()

                        # 分批次拼接到 final_key 和 final_value
                        # print("i:{}".format(i))
                        # print("index:{}".format(index))
                        del list(c[i])[0]
                        del list(c[i])[1]
                        #torch.cuda.empty_cache()
                        final_key = torch.cat((final_key, current_key), dim=2)#.contiguous()
                        final_value = torch.cat((final_value, current_value), dim=2)#.contiguous()

                        del current_key, current_value
            
                        torch.cuda.empty_cache()  # 清空显存（可选，根据具体情况判断是否必要）



                    # 将拼接结果保存到 combine_past_key_values 中
                    combine_past_key_values.append((final_key, final_value))
                    del final_key, final_value
                    torch.cuda.empty_cache()

                # 转换为元组（如果需要保持原始形式）
                combine_past_key_values = tuple(combine_past_key_values)

        del past_lst
        torch.cuda.empty_cache()
        return combine_past_key_values
    else:
        # 多个窗口的情况,如何处理
        # 去掉每个部分的query_len
        n_layers = len(past_lst[0].key_cache)
        for j in range(len(past_lst)):
            for i in range(n_layers):
                past_lst[j].key_cache[i] = past_lst[j].key_cache[i][:,:,:-query_len[j],:]
                past_lst[j].value_cache[i] = past_lst[j].value_cache[i][:,:,:-query_len[j],:]
        return past_lst
        
def generate_uniform_vector_torch(lower, upper, seq_len, before):
        # 生成取值范围内的整数
        values = torch.arange(lower, upper + 1)
        num_values = values.numel()
        
        # 计算每个值的基础重复次数和余数
        base_repeat = seq_len // num_values
        remainder = seq_len % num_values
        
        # 初始化每个值的重复次数
        repeats = torch.full((num_values,), base_repeat, dtype=torch.long)
        
        # 将余数分配给前面的几个值
        if remainder > 0:
            if before:
                repeats[:remainder] += 1
            else:
                repeats[-remainder:] += 1    
        # 根据重复次数生成向量
        vector = torch.repeat_interleave(values, repeats)
        
        return vector

def generate_pcw_position_ids(attention_mask: torch.Tensor, max_window_size: int,
                              past_key_values: Tuple[Tuple[torch.Tensor]],
                              sum_windows_size: int, windows_key_values: Tuple[Tuple[torch.Tensor]],
                              interval: float, interval_shift: int,
                              key_no_rope:bool=False,position_dict:Dict=None,
                              ) -> torch.Tensor:
    if interval_shift == 0 or interval_shift==112 :
        if key_no_rope:
            position_ids = torch.arange(0, (attention_mask.shape[1]+interval)*interval, interval)[:attention_mask.shape[1]].unsqueeze(0).to(attention_mask.device)
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values: # generate
                n_task_tokens = position_ids.shape[1] - sum_windows_size
                
                if position_dict["positional_sorting"].lower() == "raw" or position_dict["positional_sorting"].lower() == "none":
                    if n_task_tokens > 0:
                        position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
                elif position_dict["positional_sorting"].lower() == "idoffset":
                    if n_task_tokens > 0:
                        max_window_size = position_dict['max_window_size']
                        position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
                elif position_dict["positional_sorting"].lower() == "idreuse":
                    if n_task_tokens > 0:
                        max_window_size = position_dict["max_window_size"]
                        position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
                    position_ids = position_ids[:, -1].unsqueeze(-1)
                elif position_dict["positional_sorting"].lower() == "ntk" or position_dict["positional_sorting"].lower() == "pi" :
                    position_ids = position_ids[:, -1].unsqueeze(-1)
                else:
                    if n_task_tokens > 0 :
                        position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
                    position_ids = position_ids[:, -1].unsqueeze(-1)
                
                # print("position_ids:{}".format(position_ids))
                # assert 1==0
                position_ids = position_ids[:, -1].unsqueeze(-1)
            elif windows_key_values:  #第二次prefill
                positional_sorting = position_dict['positional_sorting']
                windows_rank=position_dict['windows_rank']
                windows_size=position_dict['window_size']
                token_prompt_size=position_dict['token_prompt_size']
                begin_num = 1 + token_prompt_size # 开始符与prompt
                boundary = [begin_num]
                boundary.extend(windows_size.cumsum(-1).tolist())
                rank_ascend = position_dict["rank_ascend"]
                if rank_ascend: #升序 rank最大的最recent
                    _ , windows_rank_new = torch.sort(windows_rank,dim=-1,descending=False)# 对windows_rank重新排序
                else: # 降序 rank最小的最recent
                    _ , windows_rank_new = torch.sort(windows_rank,dim=-1,descending=True)# 对windows_rank重新排序
                    windows_rank = windows_rank.max() - windows_rank
                if positional_sorting.lower() =="ntk":# 完全按照顺序即可
                    windows_size[0]-=begin_num
                    windows_size_new = windows_size[windows_rank_new]
                    new_boundary = [begin_num]
                    new_boundary.extend((windows_size_new.cumsum(-1)+begin_num).tolist())
                    for i in range(len(boundary)-1):# 对position 进行赋值
                        position_ids[:,boundary[i]:boundary[i+1]] = torch.arange(new_boundary[windows_rank[i]],new_boundary[windows_rank[i]+1]).unsqueeze(0).to(attention_mask.device)
                    # print("windows_rank_new:{}".format(windows_rank_new))
                    # 对window_size重排序
                    # torch.set_printoptions(threshold=float('inf'))
                    # print("position_ids:{}".format(position_ids))
                    # assert 1==0
                elif positional_sorting.lower() =="pi":# 完全按照顺序即可
                    windows_size_PI= windows_size*interval
                    windows_size_PI[0]-=interval*begin_num
                    windows_size_new_PI = windows_size_PI[windows_rank_new]
                    new_boundary = [interval*begin_num]
                    new_boundary.extend((windows_size_new_PI.cumsum(-1)+interval*begin_num).tolist())
                    for i in range(len(boundary)-1):# 对position 进行赋值
                        position_ids[:,boundary[i]:boundary[i+1]] = torch.arange(new_boundary[windows_rank[i]],new_boundary[windows_rank[i]+1]+interval,interval)[:boundary[i+1]-boundary[i]].unsqueeze(0).to(attention_mask.device)
                    # torch.set_printoptions(threshold=float('inf'),sci_mode=False)
                    # print("new_boundary:{}".format(new_boundary))
                    # print("position_ids:{}".format(position_ids))
                    # print("interval:{}".format(interval))
                elif positional_sorting.lower() =="idoffset":
                    offset = position_dict["offset"]
                    if offset == 4000: # 保证每个窗口都偏移 即右对齐到最大值
                        windows_rank=windows_rank+1
                    n_task_tokens = position_ids.shape[1] - sum_windows_size
                    position_ids_max = position_ids[:,:max_window_size].clone() # 获得最大编码
                    # 获得最大偏移
                    raw_model_max_len = position_dict["raw_model_max_len"]-n_task_tokens
                    # 开始偏移
                    windows_position_max = []
                    for i in range(len(boundary)-1):
                        max_offset = max(0,raw_model_max_len-windows_size[i])
                        new_offset = min(windows_rank[i]*offset,max_offset)
                        # print("new_offset:{}".format(new_offset))
                        position_ids[:,boundary[i]:boundary[i+1]] = position_ids_max[:,begin_num:boundary[i+1]-boundary[i]+begin_num] \
                            + new_offset # 可以控制offset的量
                        windows_position_max.append(windows_size[i]+begin_num+new_offset)
                    # 对input进行赋值
                    max_window_size_new = max(windows_position_max)+1
                    position_ids[:,-n_task_tokens:] = torch.arange(max_window_size_new, max_window_size_new + n_task_tokens, interval)
                    # window_size_new = [window_size+ min(windows_rank[i]*offset,max_offset) for i,window_size in enumerate(windows_size)]
                    position_dict['max_window_size'] = max_window_size_new
                    # print("max_window_size_new+n_task_tokens:{}".format(max_window_size_new+n_task_tokens))
                    # torch.set_printoptions(threshold=float('inf'),sci_mode=False)
                    # print("position_ids:{}".format(position_ids))
                    # assert 1==0
                    # print("position_dict['max_window_size']:{}".format(position_dict['max_window_size']))
                elif positional_sorting.lower() =="idreuse":
                    # 拆分id
                    factor = position_dict["factor"]
                    before = True
                    if factor == 0:
                        n_task_tokens = position_ids.shape[1] - sum_windows_size
                        raw_model_max_len = position_dict["raw_model_max_len"]-n_task_tokens
                        n_windows = len(windows_rank)
                        per_window_len = raw_model_max_len // n_windows
                        boundary_down = [per_window_len*i for i in range(n_windows)]
                        boundary_up = [per_window_len*(i+1) for i in range(n_windows)]
                        boundary_up[-1] = raw_model_max_len
                    elif factor == 1:
                        n_task_tokens = position_ids.shape[1] - sum_windows_size
                        raw_model_max_len = position_dict["raw_model_max_len"]-n_task_tokens
                        n_windows = len(windows_rank)
                        per_window_len = raw_model_max_len // n_windows
                        boundary_down = [0 for i in range(n_windows)]
                        boundary_up = [per_window_len*(i+1) for i in range(n_windows)]
                        boundary_up[-1] = raw_model_max_len
                        before = False
                    elif factor == 2:
                        n_task_tokens = position_ids.shape[1] - sum_windows_size
                        raw_model_max_len = position_dict["raw_model_max_len"]-n_task_tokens
                        n_windows = len(windows_rank)
                        per_window_len = raw_model_max_len // n_windows
                        boundary_down = [per_window_len*i for i in range(n_windows)]
                        boundary_up = [raw_model_max_len for i in range(n_windows)]
                        boundary_up[-1] = raw_model_max_len
                        before = False
                    boundary_down[0]+=begin_num
                    # print("boundary_down:{}".format(boundary_down))
                    # print("boundary_up:{}".format(boundary_up))
                    windows_size[0]-=begin_num #第一个为开始符号
                    # raw_model_max_len = 0
                    for i in range(n_windows):
                        # rank越高,下限越高
                        position_ids_temp = generate_uniform_vector_torch(boundary_down[windows_rank[i]],boundary_up[windows_rank[i]],windows_size[i],before=before).unsqueeze(0).to(attention_mask.device)
                        # raw_model_max_len_item = position_ids_temp.max().item()
                        # if raw_model_max_len_item > raw_model_max_len:
                        #     raw_model_max_len = raw_model_max_len_item
                        # print("position_ids_temp:{}".format(position_ids_temp))
                        position_ids[:,boundary[i]:boundary[i+1]] = position_ids_temp
                    max_window_size = torch.max(position_ids[:,:-n_task_tokens]).item()
                    position_ids[:,-n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
                    position_dict['max_window_size'] = max_window_size
                    # torch.set_printoptions(threshold=float('inf'))
                    # print("max_window_size:{}".format(max_window_size))
                    # print("boundary_down:{}".format(boundary_down))
                    # print("boundary_up:{}".format(boundary_up))
                    # print("position_ids.shape:{}".format(position_ids.shape))
                    # print("position_ids:{}".format(position_ids)) 
                    # assert 1==0  
                elif positional_sorting.lower() =="inwindow":
                    # 或者每个窗口的注意力分数
                    windows_recent_attn = position_dict['recent_attn'] # 1*32*size
                    windows_recent_attn_new = [ item[:,:,begin_num:].sum(0).sum(0) for item in windows_recent_attn]
                    windows_size[0]-=begin_num
                    chunk_size = position_dict['chunk_size']
                    # 根据recent attn进行块内排序
                    for i in range(len(boundary)-1):
                        # chunk_nums
                        chunk_nums = windows_size[i] // chunk_size
                        if chunk_nums*chunk_size < windows_size[i]:
                            chunk_nums += 1
                        chunk_size_all = [chunk_size for _ in range(chunk_nums)]
                        chunk_size_all[-1] =( windows_size[i] - chunk_size*(chunk_nums-1)).item()
                        chunk_size_all_temp = [0]
                        chunk_size_all_temp.extend(chunk_size_all)
                        chunk_size_sum = torch.tensor(chunk_size_all_temp).to(attention_mask.device).cumsum(-1)
                        # print("chunk_size_sum:{}".format(chunk_size_sum))
                        window_recent_attn = windows_recent_attn_new[i]
                        attn_scores = []
                        for j in range(chunk_nums):
                            attn_scores.append((torch.sum(window_recent_attn[chunk_size_sum[j]:chunk_size_sum[j+1]])/(chunk_size_sum[j+1]-chunk_size_sum[j])).item())
                        _, sorted_indices = torch.sort(torch.tensor(attn_scores), dim=-1, descending=False)
                        if not rank_ascend:
                            sorted_indices = torch.sort(torch.tensor(attn_scores), dim=-1, descending=True)
                        # 块内boundary
                        # print("sorted_indices:{}".format(sorted_indices))
                        # print("attn_scores:{}".format(attn_scores))
                        chunk_size_all_new = [chunk_size_all[sorted_indices[i]] for i in range(len(sorted_indices))]
                        # print("chunk_size_all_new:{}".format(chunk_size_all_new))
                        # print("chunk_size_all:{}".format(chunk_size_all))
                        chunk_size_all_new_temp = [0]
                        chunk_size_all_new_temp.extend(chunk_size_all_new)
                        chunk_size_sum_new = torch.tensor(chunk_size_all_new_temp).to(attention_mask.device).cumsum(-1)
                        # print("chunk_size_sum_new:{}".format(chunk_size_sum_new))
                        for j in range(chunk_nums):
                            position_ids[:,boundary[i]+chunk_size_sum[sorted_indices[j]]:boundary[i]+chunk_size_sum[sorted_indices[j]+1]] = \
                                torch.arange(begin_num+chunk_size_sum_new[j],begin_num+chunk_size_sum_new[j+1]).unsqueeze(0).to(attention_mask.device)
                        # torch.set_printoptions(threshold=float('inf'))
                        # print("boundary_down_chunk_new:{}".format(boundary_down_chunk_new))
                        # print("boundary_up_chunk_new:{}".format(boundary_up_chunk_new))
                        # print("position_ids:{}".format(position_ids[:,boundary[i]:boundary[i+1]]))
                        # assert 1==0
                    n_task_tokens = position_ids.shape[1] - sum_windows_size
                    position_ids[:,-n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
                    # torch.set_printoptions(threshold=float('inf'))
                    # print("position_ids:{}".format(position_ids))
                    # print(f"chunk_size:{chunk_size}")
                    # print(f"windows_size[{i}]:{windows_size[i]}")
                    # print(f"windows_recent_attn[{i}].shape:{windows_recent_attn_new[i].shape}")
                    # assert 1==0
                    # pass
                else:# 原始模式
                    n_task_tokens = position_ids.shape[1] - sum_windows_size
                    if n_task_tokens > 0:
                        position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids_max = position_ids[:,:max_window_size].clone() # 获得最大编码
                    # position_ids[:,0:1] = position_ids_max[:,0:1]
                    for i in range(len(boundary)-1):
                        position_ids[:,boundary[i]:boundary[i+1]] = position_ids_max[:,begin_num:boundary[i+1]-boundary[i]+begin_num]
                    # torch.set_printoptions(threshold=float('inf'))
                    # print("position_ids:{}".format(position_ids))        
        else:
            # position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = torch.arange(0, (attention_mask.shape[1]+interval)*interval, interval)[:attention_mask.shape[1]].unsqueeze(0).to(attention_mask.device)
            n_task_tokens = position_ids.shape[1] - sum_windows_size
            # print("n_task_tokens:{}".format(n_task_tokens))
            if n_task_tokens > 0:
                position_ids[0, -n_task_tokens:] = torch.arange(max_window_size*interval, (max_window_size + n_task_tokens+1)*interval, interval)[:n_task_tokens]
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values: # generate
                position_ids = position_ids[:, -1].unsqueeze(-1)
            elif windows_key_values:  #第二次prefill
                position_ids = position_ids[:, sum_windows_size:]    
            # torch.set_printoptions(threshold=float('inf'))    
            # print("position_ids:{}".format(position_ids))
    else:
        position_len = attention_mask.shape[1]
        n_task_tokens = position_len - sum_windows_size
        # print("n_task_tokens",n_task_tokens)
        if n_task_tokens > 0 :
            position_ids_new = torch.arange(max_window_size, max_window_size + (n_task_tokens+1)*interval, interval)[:n_task_tokens].unsqueeze(0).to(attention_mask.device)
        if past_key_values:
            position_ids = position_ids_new[:,-1:]
        elif windows_key_values:
            position_ids = position_ids_new
        
        # print("position_ids:{}".format(position_ids))
    return position_ids


def generate_pcw_position_ids_gemma(attention_mask: torch.Tensor, max_window_size: int,
                              past_key_values: Tuple[Tuple[torch.Tensor]],
                              sum_windows_size: int, windows_key_values: Tuple[Tuple[torch.Tensor]],
                              interval: float, interval_shift: int,
                              key_no_rope:bool=False,label:Dict=None,
                              ) -> torch.Tensor:
    if interval_shift == 0 or interval_shift==112 :
        # position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = torch.arange(0, (attention_mask.shape[1]+interval)*interval, interval)[:attention_mask.shape[1]].unsqueeze(0).to(attention_mask.device)
        n_task_tokens = position_ids.shape[1] - sum_windows_size
        if n_task_tokens > 0:
            position_ids[0, -n_task_tokens:] = torch.arange(max_window_size*interval, (max_window_size + n_task_tokens+1)*interval, interval)[:n_task_tokens]
        position_ids.masked_fill_(attention_mask == 0, 1)
        if label: # generate
            position_ids = position_ids[:, -1].unsqueeze(-1)
        else:  #第二次prefill
            position_ids = position_ids[:, sum_windows_size:]    

    return position_ids

class PCWModelWrapper:
    def __init__(self,
                 model: PreTrainedModel,tokenizer: PreTrainedTokenizerBase,device: str,context_window_size: int,
                 # base parameters
                 model_name="",
                 prompt_method: str = None,Truncation_Method: str=None,parallel_pattern:str=None,
                 raw_model_max_len:int = 3950, special_token:bool=True,
                 delete_context_prompt:bool = False,context_prompt:dict=None,
                 # window parameters
                 del_val:bool=False,rank_windows:bool=False,topk_windows:int=1,rank_cache:bool=False,
                 n_windows: int=1,window_ascend:bool=False,query_rank:bool=False,
                 decomposition_factor:int = 1,get_recent_attn:bool=False,recent_top_num:int=0,
                 query_recent_tokens:int=0,
                 # other parameters
                 right_indentation: bool = False,parallel_decoding:bool=False,NTK_aware:bool=False,
                 # kv cache eviction parameters
                 capacity:int=0,kv_cache_eviction:bool=False,
                 stage_eviction:bool=False,dynamic_window:bool=False,
                 # position shift
                 position_shift:bool=False,shift_factor:int=2,
                 interval_shift:int=0,
                 # positional sorting
                 positional_sorting:str=None,
                 rank_ascend:bool=False,
                 # attention calibration
                 input_stitching:bool=False,calibration_stage:str=None,calibration_mode:int=0,
                 ):
        # attntion calibration         
        self.input_stitching = input_stitching
        self.calibration_stage = calibration_stage
        self.calibration_mode = calibration_mode
        self.stream = torch.cuda.Stream()
        # positional sorting
        self.positional_sorting = positional_sorting
        self.rank_ascend = rank_ascend
        # base parameters
        self.model = model
        self.raw_model_max_len = raw_model_max_len
        self.tokenizer = tokenizer
        self.prompt_method = prompt_method
        self.parallel_pattern= parallel_pattern
        self.Truncation_Method = Truncation_Method
        self.device = device
        self.NTK_aware=NTK_aware
        self.special_token = special_token
        self.delete_context_prompt=delete_context_prompt
        self.context_prompt=context_prompt
        # position shift
        self.position_shift=position_shift
        self.shift_factor = shift_factor
        self.interval_shift = interval_shift
        # others
        self.parallel_decoding = parallel_decoding
        self.right_indentation = right_indentation
        # kv cache eviction
        self.kv_cache_eviction = kv_cache_eviction
        self.dynamic_window = dynamic_window
        self.capacity = capacity
        self.stage_eviction = stage_eviction
        # dynamic window
        self.del_val = del_val
        self.rank_windows = rank_windows
        self.topk_windows = topk_windows
        self.rank_cache = rank_cache
        self.window_ascend=window_ascend
        self.query_rank = query_rank
        self.query_recent_tokens = query_recent_tokens
        self.get_recent_attn = get_recent_attn
        self.decomposition_factor = decomposition_factor
        self.recent_top_num=recent_top_num
        self.model_name= model_name
        if (self.input_stitching and self.positional_sorting == "chunkllama") or "chunkllama" in self.parallel_pattern:
            from chunkllama_attn_replace import replace_with_chunkllama
            replace_with_chunkllama(pretraining_length=4096) 
            if "llama-3" in model_name.lower(): 
                logger.info("llama-3")
                self.model2 = AutoModelForCausalLM.from_pretrained("/home/avnet/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct", 
                                                 attn_implementation="eager", 
                                                 trust_remote_code=True, 
                                                 torch_dtype=torch.bfloat16).to("cuda")
            else:
                logger.info("llama2")
                self.model2 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                                 attn_implementation="eager", 
                                                 trust_remote_code=True, 
                                                 torch_dtype=torch.bfloat16).to("cuda")    
        
        if prompt_method == "complex_cot" or prompt_method == "complex_cot_pcw" \
        or self.prompt_method =="complex_cot_pcw_pre_process_window_cache" \
        or self.prompt_method == "complex_cot_pcw_multi_windows" or self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.context_window_size = context_window_size
        self.n_windows = n_windows
        # Left indentation is the default behavior as explained in the paper.
    
    def nll_loss_all(self, logits, tokenized_example):
        # logger.info("logits shape: {}".format(logits.shape))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = (tokenized_example[..., 1:]).contiguous()
        pad_token_id = self.tokenizer.pad_token_id
        # entry.labels is already padded with pad_token_id, we further pad it to full length
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)        
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        
        # 输入token的总长度
        answer_lens = (shift_labels != pad_token_id).sum(-1)
        # logger.info("answer_lens: {}".format(answer_lens))
        # logger.info("loss.shape 0: {}".format(loss.shape))
        all_loss = loss.sum(-1) / (answer_lens - 1)
        
        loss = all_loss.cpu().detach().numpy().tolist()
        return loss
    
    def nll_loss_query(self, logits, tokenized_example, query_len):
        # logger.info("logits shape: {}".format(logits.shape))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits_query = shift_logits[:,-query_len:,:] # 只取query部分
        shift_labels = (tokenized_example['input_ids'][..., 1:]).contiguous()
        shift_labels_query = shift_labels[:,-query_len:] # 只取query部分
        
        pad_token_id = self.tokenizer.pad_token_id
        # entry.labels is already padded with pad_token_id, we further pad it to full length
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
        
        
        loss = loss_fct(shift_logits_query.view(-1, shift_logits_query.size(-1)),
                    shift_labels_query.view(-1)).view(shift_labels_query.size())
        
        # 输入token的总长度
        answer_lens = (shift_labels_query != pad_token_id).sum(-1)
        # logger.info("answer_lens: {}".format(answer_lens))
        # logger.info("loss.shape 0: {}".format(loss.shape))
        all_loss = loss.sum(-1) / (answer_lens - 1)
        
        loss = all_loss.cpu().detach().numpy().tolist()
        return loss
    
    def _rank_windows(self, windows):

        for window in windows:
            logits = window['output'].logits
            tokenized_example = window['encoded_input']
            # logger.info("tokenized_example['input_ids']: {}".format(tokenized_example['input_ids'].shape))
            loss = self.nll_loss_all(logits, tokenized_example['input_ids'])[0]
            # logger.info("loss: {}".format(loss))
            #assert 1==0
            window['score'] = loss
        
        if self.rank_windows and self.rank_cache:
            if self.topk_windows < 0:
                new_windows = sorted(windows, key=lambda x: x['score'], reverse=True)[self.topk_windows:]
            else:
                new_windows = sorted(windows, key=lambda x: x['score'], reverse=True)[:self.topk_windows]
            
        elif self.rank_windows and not self.rank_cache:
            # 1. 按分数降序筛选出得分最高的 topk 窗口
            if self.topk_windows < 0:
                top_windows = sorted(windows, key=lambda x: x['score'], reverse=True)[self.topk_windows:]
            else:
                top_windows = sorted(windows, key=lambda x: x['score'], reverse=True)[:self.topk_windows]

            # 2. 用集合记录得分最高窗口的唯一标识（这里直接使用字典对象的 id）
            top_ids = {id(w) for w in top_windows}
            # window_ids = {id(w) for w in windows}
            # logger.info("top_ids: {}".format(top_ids))
            # logger.info("window_ids: {}".format(window_ids))
            # assert 1==0
            # 3. 按原始顺序过滤，只保留得分最高的窗口
            new_windows = [w for w in windows if id(w) in top_ids]
            
            del windows
            torch.cuda.empty_cache()
        if self.window_ascend:
            new_windows = new_windows.reverse()
        return new_windows

    def get_position_ids(self,max_window_size,ids,now_len):
        # max_window_size 最大的数量
        per_window_len = max_window_size // self.n_windows # 平均每个窗口可以覆盖的position ids
        
        if self.shift_factor < 0:
            interval = 1
            # boundary_down and boundary_up
            if self.shift_factor == -1: # 最大值坍缩，均摊在前面
                unit=per_window_len//self.n_windows
                boundary_down = [0 for i in range(self.n_windows)]
                boundary_up = [max_window_size-unit*(self.n_windows-i) for i in range(self.n_windows)]
                before=True
            elif self.shift_factor == -2: # 右对齐
                unit=0
                boundary_down = [0 for i in range(self.n_windows)]
                boundary_up = [max_window_size-unit*(self.n_windows-i) for i in range(self.n_windows)]
                before=False
            elif self.shift_factor == -3: # 最大值坍缩,均摊在后面
                unit=per_window_len//self.n_windows
                boundary_down = [0 for i in range(self.n_windows)]
                boundary_up = [max_window_size-unit*(self.n_windows-i) for i in range(self.n_windows)]
                before=False
            elif self.shift_factor == -4: # 最小值坍缩 坍缩在后面
                unit=per_window_len//self.n_windows
                boundary_down = [0+i*unit for i in range(self.n_windows)]
                boundary_up = [now_len for i in range(self.n_windows)]
                before=False
            elif self.shift_factor == -5: # 最小值坍缩 坍缩在前面
                unit=per_window_len//self.n_windows
                boundary_down = [0+i*unit for i in range(self.n_windows)]
                boundary_up = [now_len for i in range(self.n_windows)]
                before=True
            boundary_up[-1] = max_window_size
            results = generate_uniform_vector_torch(boundary_down[ids],boundary_up[ids],now_len,before).unsqueeze(0).to(self.device)
            max_position_idx = now_len
        if self.interval_shift > 0: 
            # 为每个窗口分配不同的间隔
            begin = 0
            if self.interval_shift == 1:
                interval = 1.0
            elif self.interval_shift == 2 or self.interval_shift == 6:
                interval = 1 / (ids*0.1+1)
            elif self.interval_shift == 3:
                interval = 1 / (ids*0.05+1)
            elif self.interval_shift == 4:
                interval = 1 / (ids*0.15+1)
            elif self.interval_shift == 5: # 每次平移0.5个位置
                interval = 1.0
                begin = 0 + 0.5*ids
            interval = min(interval,1)
            results = torch.arange(begin,(now_len+1+begin)*interval,interval).unsqueeze(0).to(self.device)
            results = results[:,:now_len]
            max_position_idx = results[:,-1].item() +interval
            # max_position_idx = math.ceil(results[:,-1]+interval)
        # print("max_position_idx:{}".format(max_position_idx))
        # print("results.shape:{}".format(results.shape))
        # print("results:{}".format(results))
        
        return interval,max_position_idx,results
    
    def apply_qwen2(self, truncation_prompts):
        if  truncation_prompts == "\n":
            return truncation_prompts
        if "qwen2" in self.model_name.lower() :
            messages = [
                    {"role": "user", "content": truncation_prompts}
            ]
            if "qwen2" in self.model_name.lower():
                truncation_prompts = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
            )
        elif "gemma" in self.model_name.lower():
            # truncation_prompts = f"<start_of_turn>user\n{truncation_prompts}<end_of_turn>\n<start_of_turn>model\n"
            # truncation_prompts = self.tokenizer.decode(truncation_prompts, skip_special_tokens=False)
            pass
        elif "llama-3" in self.model_name.lower():
            # truncation_prompts = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{truncation_prompts}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
            pass
        return truncation_prompts
    
    def get_window(self,rank, text , raw_location:int,
                   max_window_size:int, raw_model_max_len:int, context_max_len:int, 
                   recent_token:int,first_max_len:int
                   ):
        if raw_location == 0:
            for i in range(len(self.model.model.layers)):
                self.model.model.layers[i].self_attn.token_prompt_size = 0
        else:
            for i in range(len(self.model.model.layers)):
                self.model.model.layers[i].self_attn.token_prompt_size = self.token_prompt_size + 1
        
        
        if "anchor_new" in self.parallel_pattern:
                # 传入字典
            encoded_input_window_context = self.tokenizer(text['context'],
                                          padding="longest", 
                                          return_tensors="pt", 
                                          add_special_tokens=True,
                                          ).to(self.device)
            encoded_input_window_query = self.tokenizer(text['input'],
                                          padding="longest", 
                                          return_tensors="pt", 
                                          add_special_tokens=True,
                                          ).to(self.device)
            len_context=encoded_input_window_context.input_ids.shape[1]
            len_query=encoded_input_window_query.input_ids.shape[1]
            # logger.info("len_context: {}".format(len_context))
            # logger.info("len_query: {}".format(len_query))
            encoded_input_window = {
                "input_ids":
                    torch.cat([encoded_input_window_context["input_ids"],encoded_input_window_query["input_ids"]],dim=1),
                "attention_mask":
                    torch.cat([encoded_input_window_context["attention_mask"],encoded_input_window_query["attention_mask"]],dim=1),
                }
        else:
            # logger.debug(f"text is : {text}")
            # 输入的context可能包括query
            text = self.apply_qwen2(text)
            encoded_input_window = self.tokenizer(text,
                                          padding="longest", 
                                          return_tensors="pt", 
                                          add_special_tokens=True,
                                          ).to(self.device)
        # logger.debug(f"text len is encoded_input_window[input_ids].shape: {encoded_input_window['input_ids'].shape}")
#           # 如果添加了query，希望context+query不超过一半
        if "anchor_new" in self.parallel_pattern:
            if self.NTK_aware: # 不再截断
                pass
            else:
                if len_context > first_max_len and "default" not in self.parallel_pattern:
                    half = int(first_max_len/2)
                    
                    truncation_prompts = [self.tokenizer.decode(encoded_input_window_context['input_ids'][i][:half], skip_special_tokens=False)+
                              self.tokenizer.decode(encoded_input_window_context['input_ids'][i][-half:], skip_special_tokens=False) 
                              for i in range(len(encoded_input_window_context['input_ids']))]
                    truncation_prompts = self.apply_qwen2(truncation_prompts[0])
                    truncation_prompts=[truncation_prompts]
                    encoded_input_window_context = self.tokenizer(truncation_prompts, 
                                                       padding="longest", 
                                                       return_tensors="pt", 
                                                       add_special_tokens=True).to(self.device)
                    # len_query = recent_token
                    # print("len_query:{}".format(len_query))
                    # print("first_max_len",first_max_len)
                    # print("first_max_len-context_max_len",first_max_len-context_max_len)
                    len_query = min(raw_model_max_len-first_max_len-1,len_query)
                    # print("len_query:{}".format(len_query))
                    # print("raw_model_max_len-context_max_len",raw_model_max_len-context_max_len)
                    assert len_query <= raw_model_max_len-context_max_len 
                elif len_context+len_query > first_max_len and "default" and "default" not in self.parallel_pattern:
                    # len_query = first_max_len-len_context+1
                    len_query = min(raw_model_max_len-len_context-1,len_query)
                    
                encoded_input_window = {
                    "input_ids":
                        torch.cat([encoded_input_window_context["input_ids"],encoded_input_window_query["input_ids"][:,-len_query:]],dim=1),
                    "attention_mask":
                        torch.cat([encoded_input_window_context["attention_mask"],encoded_input_window_query["attention_mask"][:,-len_query:]],dim=1),
                }
                # logger.info("len_context: {}".format(encoded_input_window_context['input_ids'].shape[1]))
                # logger.info("len_query: {}".format(len_query))
        else:
            if self.NTK_aware:
                pass
            else:
                if encoded_input_window['input_ids'].shape[1] > first_max_len and "default" not in self.parallel_pattern:
                    half = int(first_max_len/2)
                    
                    truncation_prompts = [self.tokenizer.decode(encoded_input_window.input_ids[i][:half], skip_special_tokens=False)+
                              self.tokenizer.decode(encoded_input_window.input_ids[i][-half:], skip_special_tokens=False) 
                              for i in range(len(encoded_input_window.input_ids))]
                    truncation_prompts = self.apply_qwen2(truncation_prompts[0])
                    truncation_prompts=[truncation_prompts]
                    encoded_input_window = self.tokenizer(truncation_prompts, 
                                                       padding="longest", 
                                                       return_tensors="pt", 
                                                       add_special_tokens=True).to(self.device)

        if  "anchor" in self.parallel_pattern:     
            if "anchor_new" in self.parallel_pattern : 
                query_len = len_query
                if not self.stage_eviction:
                    for i in range(len(self.model.model.layers)):
                        self.model.model.layers[i].self_attn.window_size = recent_token
                        self.model.model.layers[i].self_attn.query_len = query_len
                        self.model.model.layers[i].self_attn.capacity = min(self.capacity,len_context)+query_len
                else:
                    for i in range(len(self.model.model.layers)):
                        self.model.model.layers[i].self_attn.window_size = recent_token
                        self.model.model.layers[i].self_attn.query_len = query_len
                        self.model.model.layers[i].self_attn.capacity = min(self.capacity,len_context)
                        # logger.debug(f"self.model.model.layers[i].self_attn.capacity {self.model.model.layers[i].self_attn.capacity}")
            else:
                target_token_id = self.tokenizer.convert_tokens_to_ids('[Anchor]')
                
                positions = torch.where(encoded_input_window["input_ids"][0] == target_token_id)[0]
                if positions.shape[0] == 0:
                    torch.set_printoptions(threshold=float('inf'))
                    print("target_token_id:{}".format(encoded_input_window["input_ids"][0]))
                    print("text: {}".format(text))
                    assert 1==0
                query_len = len(encoded_input_window["input_ids"][0,positions[-1]:])
                for i in range(len(self.model.model.layers)):
                    # 确定window_size 可以根据query进行选择
                    # window_size == query_len 则根据query_len进行筛选
                    self.model.model.layers[i].self_attn.window_size = recent_token
                    self.model.model.layers[i].self_attn.query_len = query_len
                    self.model.model.layers[i].self_attn.capacity = self.capacity+query_len
        else:
            query_len = 0
        # logger.info("positions: {}".format(positions))
        # logger.info("query_len: {}".format(query_len))
        
        window_size = encoded_input_window['input_ids'].shape[1]
        # print("window_size:{}".format(window_size)) #encoded_input_window["input_ids"]
        # print("encoded_input_window[input_ids].shape:{}".format(encoded_input_window["input_ids"].shape))
        
        if self.right_indentation:
            shift = max_window_size - window_size
            encoded_input_window["position_ids"] = encoded_input_window["attention_mask"].cumsum(-1) - 1 + shift
        # print("encoded_input_window[position_ids]:{}".format(encoded_input_window["position_ids"]))
        max_position_idx = encoded_input_window["attention_mask"].shape[1]
        interval = 1
        if self.position_shift:
            if self.shift_factor < 0 or self.interval_shift > 0:
                interval,max_position_idx,encoded_input_window["position_ids"] = self.get_position_ids(max_window_size,raw_location,
                                                                         encoded_input_window["attention_mask"].shape[1])
            else:
                shift = raw_location * self.shift_factor
                encoded_input_window["position_ids"] = encoded_input_window["attention_mask"].cumsum(-1) - 1 + shift
                max_position_idx = max_position_idx + shift
                interval = 1
                # 分配position_ids
        
        with torch.no_grad():
            # 在这里完成上下文窗口的编码, prefill
            if "default" in self.parallel_pattern:
                context_length = encoded_input_window['input_ids'].shape[1]
                if"chunkllama" in self.parallel_pattern:
                    # assert 1==0
                    if "ppl" in self.parallel_pattern:
                        output = self.model2(**encoded_input_window,labels=encoded_input_window['input_ids'])
                        ppl = torch.exp(output['loss'])
                        print("ppl:{}".format(ppl))
                        assert 1==0
                    else:
                        res = self.model2.generate(**encoded_input_window,
                                                num_beams=1,
                                                do_sample=False,
                                                temperature=1.0,
                                                min_length=context_length+1,
                                                max_new_tokens=64,
                                                )[0]
                        res = self.tokenizer.decode(res[context_length:], skip_special_tokens=True)
                        print(f"res is :{res}")
                        # assert 1==0
                    return res
                    
                elif "PI" in self.parallel_pattern:
                    if "ppl" in self.parallel_pattern:
                        interval = 4096 /  encoded_input_window['input_ids'].shape[1] 
                        if interval > 1:
                            interval = 1
                        encoded_input_window['position_ids'] = torch.arange(0, (encoded_input_window['input_ids'].shape[1]+1)*interval, interval)[:encoded_input_window['input_ids'].shape[1]].unsqueeze(0).to(self.device)
                        print("encoded_input_window['input_ids'].shape:{}".format(encoded_input_window['input_ids'].shape))
                        output = self.model(**encoded_input_window,labels=encoded_input_window['input_ids'])
                        ppl = torch.exp(output['loss'])
                        print("ppl:{}".format(ppl))
                        assert 1==0
                    else:
                        interval = self.raw_model_max_len /  encoded_input_window['input_ids'].shape[1] 
                        logger.info(f"interval: {interval}")
                        if interval > 1:
                            interval = 1
                        encoded_input_window['position_ids'] = torch.arange(0, (encoded_input_window['input_ids'].shape[1]+1)*interval, interval)[:encoded_input_window['input_ids'].shape[1]].unsqueeze(0).to(self.device)
                        output = self.model(**encoded_input_window)
                else:
                    if "ppl" in self.parallel_pattern:
                        with torch.no_grad():
                            print("encoded_input_window['input_ids'].shape:{}".format(encoded_input_window['input_ids'].shape))
                            output = self.model(**encoded_input_window)
                            del output.past_key_values
                            del output.attentions
                            torch.cuda.empty_cache()
                            logits = output['logits']
                            labels = encoded_input_window['input_ids']
                            
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            shift_logits = shift_logits.view(-1, self.model.model.config.vocab_size)
                            shift_labels = shift_labels.view(-1)
                            torch.cuda.empty_cache()
                            loss = F.cross_entropy(shift_logits, shift_labels)
                            ppl = torch.exp(loss)
                            
                            ppl = torch.exp(output['loss'])
                            print("ppl:{}".format(ppl))
                            assert 1==0
                    else:
                        from modeling_gemma_with_pcw_kv_cache_FlashAttention_longbench import Gemma2ForCausalLMPCW
                        if isinstance(self.model,Gemma2ForCausalLMPCW):
                            
                            past_key_values = []
                            output = self.model(**encoded_input_window,use_cache=True,past_key_values=past_key_values)
                            # print("output:{}".format(len(output.past_key_values[0])))
                            # assert 1==0
                        else:
                            output = self.model(**encoded_input_window)
            else:
                from modeling_gemma_with_pcw_kv_cache_FlashAttention_longbench import Gemma2ForCausalLMPCW
                if isinstance(self.model,Gemma2ForCausalLMPCW):
                    if "list" not in self.parallel_pattern:
                        kv_cache = HybridCache(config=self.model.config,
                                                   max_batch_size=1,
                                                   max_cache_len=encoded_input_window["input_ids"].shape[1]-query_len,
                                                   device=self.device,
                                                   dtype=self.model.dtype,)
                    else:
                        kv_cache = []
                    output = self.model(**encoded_input_window,use_cache=True,past_key_values=kv_cache)
                else:
                    output = self.model(**encoded_input_window)
                if self.position_shift:
                    del encoded_input_window["position_ids"]
        
        return {'text': text,
                'encoded_input': encoded_input_window,
                'attention_mask': encoded_input_window['attention_mask'],
                'window_size': window_size,
                'output': output,
                'past': output['past_key_values'],
                'query_len': query_len,
                'max_position_idx': max_position_idx,
                'interval':interval
                }
    
    def _get_windows_longbench(self, texts: List[str],context_max_len:int,
                               raw_model_max_len:int,
                               recent_token:int
                               ) -> List[Dict]:
        windows = []
        """
        texts: 传过来的不同上下文:
        
        """
        if self.capacity == 112:
            self.right_indentation = True
        logger.debug(f"self.right_indentation: {self.right_indentation}")
        if self.right_indentation:
            max_window_size = max(n_tokens_in_prompt(self.tokenizer, t, add_special_tokens=True) for t in texts)
        else:
            if "anchor_new" in self.parallel_pattern:
                new_texts = [text['context']+text['input'] for text in texts]
                max_window_size = max(n_tokens_in_prompt(self.tokenizer, t, add_special_tokens=True) for t in new_texts)
            else:
                max_window_size = max(n_tokens_in_prompt(self.tokenizer, t, add_special_tokens=True) for t in texts)

        first_max_len = context_max_len+(raw_model_max_len-context_max_len)//2
        
        
        if "default" in self.parallel_pattern:
            windows_rank = []
        if not self.dynamic_window:
            for raw_location,text in enumerate(texts):
                window = self.get_window(raw_location,text,raw_location,max_window_size,raw_model_max_len,context_max_len,recent_token,first_max_len)
                if "chunkllama" in self.parallel_pattern:
                    return window,window
                windows.append(window)
                window['raw_location'] = raw_location
            if self.rank_windows:
               windows = self._rank_windows(windows)
        else:
            if self.topk_windows < 0:
                size = -self.topk_windows
            else:
                size = self.topk_windows
            pq = priorityqueue.FixedSizePriorityQueue(size=size,key="loss")
            # 统计最小的context_len
            if "anchor_new" in self.parallel_pattern:
                min_context_len = min(n_tokens_in_prompt(self.tokenizer, t['context'], add_special_tokens=False) for t in texts)
                self.capacity = min(self.capacity,min_context_len)
                # print("self.capacity:{}".format(self.capacity))

            default_stream = torch.cuda.current_stream()
 
            torch.cuda.Stream.synchronize(default_stream)
            if "stream" in self.parallel_pattern:
                my_windows = []
                for raw_location,text in enumerate(texts):
                    with torch.cuda.stream(self.stream):
                        window = self.get_window(raw_location,text,raw_location,max_window_size,raw_model_max_len,context_max_len,recent_token,first_max_len)
                    torch.cuda.current_stream().wait_stream(self.stream)
                    my_windows.append(window)
                        
            for raw_location,text in enumerate(texts):
                if "stream" in self.parallel_pattern:
                    window = my_windows[raw_location]
                    pass
                else:
                    window = self.get_window(raw_location,text,raw_location,max_window_size,raw_model_max_len,context_max_len,recent_token,first_max_len)
                
                # 对每个窗口进行操作
                logits = window['output'].logits
                tokenized_example = window['encoded_input']
                eviction_lens = []
                for layer in self.model.model.layers:
                    eviction_lens.append(layer.self_attn.eviction_len)
                # logger.info(f"eviction_lens: {eviction_lens}")
                window["eviction_len"] = eviction_lens
                
                if self.get_recent_attn:
                    if self.recent_top_num!=0: # 注意力的集中度设置为loss的值
                        if self.recent_top_num > 0:
                            recent_attn = self.model.model.layers[-1].self_attn.recent_attn
                            recent_attn = recent_attn.topk(self.recent_top_num,dim=-1).values
                            loss = torch.sum(recent_attn)/(recent_attn.shape[1]*recent_attn.shape[2])
                        else:
                            sum_attn = 0
                            for layer in self.model.model.layers:
                                layer_attn = layer.self_attn.recent_attn
                                layer_attn = layer_attn.topk(-self.recent_top_num,dim=-1).values
                                # recent 8的累积注意力分数 的 top recent_top_num token
                                sum_attn += torch.sum(layer_attn)
                                # logger.debug(f"sum_attn: {sum_attn}")
                            loss = sum_attn / (layer_attn.shape[1]*layer_attn.shape[2]*len(self.model.model.layers))    
                                # 每个窗口的注意力
                    else:
                        recent_attn = self.model.model.layers[-1].self_attn.recent_attn
                        loss = torch.sum(recent_attn)/(recent_attn.shape[1]*recent_attn.shape[2])
                    
                    if self.topk_windows < 0: #取损失最小的5个
                        loss = -loss
                    logger.info(f"loss: {loss}")
                    # assert 1==0
                else:
                    # 获得最后一层的累积注意力分数
                    recent_attn = self.model.model.layers[-1].self_attn.recent_attn
                    if self.query_rank:
                        if self.query_recent_tokens>0:
                            loss = self.nll_loss_query(logits, tokenized_example, min(self.query_recent_tokens,window['query_len']))[0]
                        else:
                            loss = self.nll_loss_query(logits, tokenized_example, window['query_len'])[0]
                    else:
                        loss = self.nll_loss_all(logits, tokenized_example['input_ids'])[0]
                    if self.topk_windows < 0: #取损失最小的5个
                        loss = -loss
                
                if self.interval_shift == 112:
                    layer_attn = self.model.model.layers[0].self_attn.recent_attn
                    for layer in self.model.model.layers[1:]:
                        layer_attn = layer.self_attn.recent_attn+layer_attn
                    layer_attn = layer_attn/len(self.model.model.layers)
                    # print("layer_attn.shape:{}".format(layer_attn.shape))
                else:    
                    layer_attn = recent_attn        
                if layer_attn!=None:   
                    window['recent_attn'] = layer_attn[:,:,:-window['query_len']]
                window['raw_location'] = raw_location
                
                # 对窗口随机赋值
                if "shuffle" in self.parallel_pattern:
                    window['loss'] = random.uniform(0,3)
                else:
                    window['loss'] = loss
                pq.add(window)
                if self.rank_windows and self.rank_cache: 
                    if not self.window_ascend: # 损失值最大的在最前面
                        if self.topk_windows < 0: 
                            windows = pq.get_elements_ascending()
                        else:
                            windows = pq.get_elements_descending()
                    else: # 值最小的在最前面
                        if self.topk_windows < 0:
                            windows = pq.get_elements_descending()
                        else:
                            windows = pq.get_elements_ascending()
                elif self.rank_windows and not self.rank_cache: # 默认顺序
                    windows = pq.get_elements_key("raw_location")
                    if "random" in self.parallel_pattern:
                        windows = pq.get_elements_random()
                #默认顺序
                windows_ids = pq.get_elements_ascending_ids()
                # print("windows_ids:{}".format(windows_ids))
                windows_rank = []
                for window in windows:
                    windows_rank.append(windows_ids.index(window['raw_location']))
                # print(f"windows: {windows}")
        logger.info(f"raw_location: {[window['raw_location'] for window in windows]}") 
        # print("texts[raw_location]:{}".format(texts[windows[0]['raw_location']]))
        if "default" in self.parallel_pattern:
            for window in windows:
                window["eviction_len"] = [0]*32
        # logger.info("len of windows in _get_windows_longbench: {}".format(len(windows)))
        return windows,windows_rank
     
    def get_contexts_cache_longbench(self, contexts: List[str],context_max_len:int,
                                     raw_model_max_len:int,
                                     recent_token:int
                                     ) -> Dict:
        """
            contexts： 喂入的不同上下文 windows_few_shots
            
            分段处理输入上下文（_get_windows 函数）：

            将多个上下文输入文本转化为模型的编码输入，同时生成模型的输出、历史状态（past_key_values）和注意力掩码。
            支持右对齐功能，确保对齐后输入长度一致。
            
            
            优化上下文处理和缓存（get_contexts_cache 函数）：

            从上下文窗口中提取关键信息（如 past_key_values 和 attention_mask），用于后续高效推理。
            合并上下文，避免重复的特殊标记（如 BOS token），并记录关键指标（如最大窗口大小）。
        """
        # print("len(contexts):{}".format(len(contexts)))
        # 实际上是有多少个上下文快
        if self.delete_context_prompt:
            # past_key_value拼接时，去掉这些token
            token_prompt_size = n_tokens_in_prompt(self.tokenizer, self.context_prompt, add_special_tokens=False)
        else:
            token_prompt_size = 0
        for i in range(len(self.model.model.layers)):
            self.model.model.layers[i].self_attn.token_prompt_size = token_prompt_size 
            self.token_prompt_size = token_prompt_size
            
        windows,windows_rank = self._get_windows_longbench(contexts,context_max_len,raw_model_max_len,recent_token)
        if "chunkllama" in self.parallel_pattern:
            return windows
        logger.debug(f"len of windows in get_contexts_cache:{len(windows)}")
        if self.decomposition_factor!=1:
            size = self.decomposition_factor*abs(self.topk_windows)
            boundary_ss = priorityqueue.FixedSizePriorityQueue(size=size,key="loss")
            for location,window in enumerate(windows):
                # 细粒度的拆分，只更改past_key_value的顺序就好，不需要重新编码
                persize = window['window_size'] // self.decomposition_factor
                per_boundaries = [persize*i for i in range(0,self.decomposition_factor)]
                if location != 0 : # 去掉开始符 只保留一个开始符
                    per_boundaries[0] += 1
                per_boundaries.append(window['window_size'])
                for i in range(self.decomposition_factor):
                    loss = self.nll_loss_all(window['output'].logits[:,per_boundaries[i]:per_boundaries[i+1],:], window['encoded_input']['input_ids'][:,per_boundaries[i]:per_boundaries[i+1]])
                    item = {"location":location,"loss":loss,"boundary":[per_boundaries[i],per_boundaries[i+1]]}
                    boundary_ss.add(item)
                        
            self.boundary_ss = boundary_ss
        

        if self.kv_cache_eviction:
            # query_len 为 window_size
            logger.debug(f"self.capacity: {self.capacity}")
            # windows_sizes = [min(self.capacity,window['window_size'])-window['query_len'] for window in windows]
            windows_sizes = [min(self.capacity,window['window_size']-window['query_len']) for window in windows]
            # window['window_size'] 包含了query_len的长度，但是要去掉query_len的长度
            capacity_add_query = [min(self.capacity,window['window_size']-window['query_len'])+window['query_len'] for window in windows]
            max_window_size = max([window['window_size']-window['query_len'] for window in windows])
            
            if "anchor" not in self.parallel_pattern: # 不裁剪query
                for window in windows:
                    window['query_len'] = -window['encoded_input'].input_ids.shape[1]

            past_attention_mask = torch.cat([windows[0]['attention_mask'][:,-capacity_add_query[0]:-windows[0]['query_len']-windows[0]['eviction_len'][0]]] + 
                            [window['attention_mask'][:, -capacity+1+token_prompt_size:-window['query_len']-window['eviction_len'][0]] 
                            for window,capacity in zip(windows[1:],capacity_add_query[1:])], dim=1)
        else:
            windows_sizes = [window['window_size']-window['query_len'] for window in windows]
            max_window_size = max(windows_sizes)
            windows_sizes = [window_size-token_prompt_size for window_size in windows_sizes]
            windows_sizes[0] += token_prompt_size
            
            if "anchor" not in self.parallel_pattern: # 不裁剪query
                for window in windows:
                    window['query_len'] = -window['encoded_input'].input_ids.shape[1]
            
            past_attention_mask = torch.cat([windows[0]['attention_mask'][:,:-windows[0]['query_len']]] + 
                        [window['attention_mask'][:, 1+token_prompt_size:-window['query_len']] 
                         for window in windows[1:]], dim=1)
            

        raw_windows_sizes = [window['window_size'] for window in windows]
        predict_token = [torch.argmax(window['output'].logits, dim=-1) for window in windows]
        # print("predict_token[0][:,-1:]:{}".format(predict_token[0][:,-1:].shape))
        # 获取所有 window 的 logits
        logits_last_token = [window['output'].logits[:, -1, :] for window in windows]
        # print("logits_last_token[0].shape:{}".format(logits_last_token[0].shape))
        # print("logits_last_token[1].shape:{}".format(logits_last_token[1].shape))
        # print("torch.stack(logits_last_token).shape:{}".format(torch.stack(logits_last_token).shape))
        # 将所有 logits 堆叠起来，计算平均值 
        # logits_last_token[0].shape:torch.Size([1, 32001])
        # logits_last_token[1].shape:torch.Size([1, 32001])
        # torch.stack(logits_last_token).shape:torch.Size([2, 1, 32001])
        # logits_avg shape:torch.Size([1, 32001])
        # if self.parallel_decoding == True:
        logits_avg = torch.stack(logits_last_token).mean(dim=0)
        last_token = torch.argmax(logits_avg, dim=-1)

        # print("self.del_val:{}".format(self.del_val))
        # assert 1==0
        max_raw_window_size = max(raw_windows_sizes)
        raw_sum_windows_size = sum(raw_windows_sizes) - (len(windows) - 1)
        sum_windows_size = sum(windows_sizes) - (len(windows) - 1)
        max_window = max(windows, key=lambda window: window['max_position_idx'])
        if self.interval_shift == 6:
            interval = min([window['interval'] for window in windows])
        else:
            interval = max([window['interval'] for window in windows])
        position_dict = {}
        position_dict['window_size'] = [window_size-1 for window_size in windows_sizes]
        position_dict['window_size'][0]+=1
        position_dict['window_size'] = torch.tensor(position_dict['window_size']).to(self.device)
        position_dict['windows_rank'] = torch.tensor(windows_rank).to(self.device)
        position_dict["positional_sorting"] = self.positional_sorting
        position_dict["rank_ascend"] = self.rank_ascend
        position_dict['raw_model_max_len'] = raw_model_max_len
        position_dict["token_prompt_size"] = token_prompt_size
        if self.positional_sorting is not None:
            if self.positional_sorting.lower() == "idoffset":
                position_dict['offset'] = self.shift_factor
            elif self.positional_sorting.lower() == "idreuse":
                position_dict['factor'] = self.shift_factor
            elif self.positional_sorting.lower() == "inwindow":
                position_dict['chunk_size'] = self.shift_factor
                position_dict['recent_attn'] = [window['recent_attn'] for window in windows]
        
        past_key_values = combine_past_key_values_longbench(self,[window['past'] for window in windows], [window['eviction_len'] for window in windows ],
                                                                     query_len=[window['query_len'] for window in windows],
                                                                     token_prompt_size=token_prompt_size, del_val=self.del_val,
                                                                     )
        # if self.delete_context_prompt:
        #     assert 1==0
        
        if self.input_stitching:
            input_stitching_dic = {}
            input_stitching_dic['input_stitching'] = torch.cat([windows[0]['encoded_input']['input_ids'][:,:-windows[0]['query_len']] ] +
                                        [window['encoded_input']['input_ids'][:,1+token_prompt_size:-window['query_len']]
                                         for window in windows[1:]], dim=1).to(self.device) 
            input_stitching_dic['position_ids'] = torch.cat([torch.arange(0,windows_sizes[0]).unsqueeze(0).to(self.device)]+
                                                            [torch.arange(1+token_prompt_size,token_prompt_size+windows_sizes[i]).unsqueeze(0).to(self.device) 
                                                             for i in range(1,len(windows))],dim=1)
        else:
            input_stitching_dic =None
        return {'past_key_values': past_key_values,
                'max_window_size': max_window_size,
                'max_raw_window_size': max_raw_window_size,
                'past_attention_mask': past_attention_mask,
                'sum_windows_size': sum_windows_size,
                'raw_sum_windows_size': raw_sum_windows_size,
                'first_token': [predict_token[i][:,-1:] for i in range(len(windows))],
                'first_token_avg': [last_token],
                'max_position_idx': max_window['max_position_idx'],
                'interval': interval,
                'position_dict': position_dict,
                'input_stitching_dic':input_stitching_dic,
                }

    def pcw_generate_longbench(self,
                               per_windows_prompt: List[str],
                               output_max_len: int,
                               parallel_patterns:str,
                               question="",
                               context_max_len=3600,
                               raw_model_max_len=3950,
                               raw_position_select:bool=True,
                               adaptive_n_windows = None,
                               recent_token:int=8,
                               **kwargs,
                               ):
        with torch.inference_mode():
            if self.prompt_method=="complex_cot_pcw_multi_windows" \
                or self.prompt_method=="complex_cot_pcw_multi_windows_kv_cache":
                cache = self.get_contexts_cache_longbench(per_windows_prompt,context_max_len=context_max_len,
                                                          raw_model_max_len=raw_model_max_len,recent_token=recent_token)
                if "gemma" not in self.model_name:
                    if "chunkllama" in self.parallel_pattern:
                        return cache
                    # logger.info(f"windows_max_token_size before input: {cache['max_window_size']}")
                    # logger.info(f"raw_windows_max_token_size before input: {cache['max_raw_window_size']}")
                    
                    logger.info(f"cache['sum_windows_size']: {cache['sum_windows_size']}")
                    logger.info(f"past_attention_mask.shape: {cache['past_attention_mask'].shape}")
                    logger.info(f"cache['past_key_values'][0][0].shape: {cache['past_key_values'][0][0].shape}")
                    # assert cache['sum_windows_size'] == cache['past_key_values'][0][0].shape[2]
                    assert cache['sum_windows_size'] == cache['past_attention_mask'].shape[1]
                    
                    input_max_window_size=cache['max_raw_window_size']
                    
                    if "every_window_query_input_no_query" in parallel_patterns and "anchor" not in parallel_patterns:
                        input = "\n"
                    elif "every_window_no_query_input_query" in parallel_patterns:
                        # assert 1==0
                        input = question
                    elif "every_window_query_input_query" in parallel_patterns and "anchor" not in parallel_patterns:
                        input = question
                    elif "anchor" in parallel_patterns:
                        input = question
                    else:
                        input = "\n"
                    logger.info(f"input is: {input}")
                    #控制位置编码位置
                    if raw_position_select:
                        input_max_window_size = cache['max_raw_window_size']
                    else:
                        input_max_window_size = cache['max_window_size']
                    
                    if self.position_shift:
                        input_max_window_size = cache['max_position_idx']
                    print("input_max_window_size:{}".format(input_max_window_size))
                    
                    special_token = self.special_token
                    first_token = cache['first_token_avg']
                    if "gemma2" in self.parallel_pattern:
                        pass
                    else:
                        input = self.apply_qwen2(input)
                    tokenized_inputs = self.tokenizer.encode_plus(input, 
                                                                  truncation = True, 
                                                                  return_tensors='pt', 
                                                                  add_special_tokens=special_token)
                    tokenized_inputs_attention_mask = tokenized_inputs.attention_mask.cuda()
                    context_length = tokenized_inputs.input_ids.shape[1]
                    tokenized_inputs = tokenized_inputs.input_ids.cuda()
                    
                    if self.parallel_decoding:
                        tokenized_inputs = torch.cat((tokenized_inputs, first_token[0].unsqueeze(0)), dim=1)
                        tokenized_inputs_attention_mask = tokenized_inputs.ne(self.tokenizer.pad_token_id)
                        # assert 1==0
                    
                    input_ids_length = tokenized_inputs.shape[1]+cache['max_window_size']
                    
                    if "default" in parallel_patterns:
                        assert len(cache['first_token']) == 1
                        # 获得第一个窗口的first_token
                        tokenized_inputs = cache['first_token'][0]
                        print("tokenized_inputs:{}".format(tokenized_inputs))
                        context_length = tokenized_inputs.shape[1]
                        tokenized_inputs = tokenized_inputs
                        tokenized_inputs_attention_mask = tokenized_inputs.ne(self.tokenizer.pad_token_id)
                        output_max_len=output_max_len-1
                    
                    #logger.info(f"A window: generate all tokens length: {tokenized_inputs.shape[1]+cache['max_window_size']}")
                    
                    # 第二次 truncation 确保添加query后仍在模型最大长度内
                    if self.NTK_aware: # 不需要截断
                        pass
                    else:
                        if input_ids_length > raw_model_max_len and "default" not in parallel_patterns:
                            logger.info("begin trucate input_ids")
                            logger.info(f"model_max_length: {raw_model_max_len}")
                            logger.info(f"input_ids_length: {input_ids_length}")
                            logger.info(f"cache['max_window_size']: {cache['max_window_size']}")
                            half = (raw_model_max_len-cache['max_window_size'])//2
                            # windows+query本身就超过了model_max_length 暂时先不处理 （可能需要考虑截断）
                            assert half > 0
                            
                            truncation_input = [self.tokenizer.decode(tokenized_inputs[i][:half], skip_special_tokens=True)+
                                  self.tokenizer.decode(tokenized_inputs[i][-half:], skip_special_tokens=True) 
                                  for i in range(len(tokenized_inputs))]
                            truncation_input = self.apply_qwen2(truncation_input[0])
                            truncation_input = [truncation_input]
                            new_tokenized_inputs = self.tokenizer(truncation_input, 
                                                                  truncation=True,
                                                                  return_tensors='pt', 
                                                                  add_special_tokens=special_token)
                            
                            tokenized_inputs_attention_mask = new_tokenized_inputs.attention_mask.cuda()
                            tokenized_inputs = new_tokenized_inputs.input_ids.cuda()
                            context_length = tokenized_inputs.shape[1]

                    logger.info(f"input tokens length: {tokenized_inputs.shape[1]}")
                    logger.info(f"A window: after truncation generate all tokens length: {tokenized_inputs.shape[1]+cache['max_window_size']}")
                    
                    sum_windows_size = cache['sum_windows_size']
                    
                    combined_attention_mask = torch.cat((cache['past_attention_mask'], tokenized_inputs_attention_mask),dim=1)
                    logger.info(f"combined_attention_mask.shape: {combined_attention_mask.shape}")
                    logger.info(f"input_max_window_size: {input_max_window_size}")
                    logger.info(f"sum_windows_size: {sum_windows_size}")
                    assert combined_attention_mask.shape[1]-sum_windows_size == tokenized_inputs_attention_mask.shape[1]
                    logger.info(f"interval: {cache['interval']}")
                    logger.info(f"interval_shift: {self.interval_shift}")
                    
                  
                    if self.positional_sorting is not None:
                        if self.positional_sorting.lower()=="pi":
                            interval = raw_model_max_len/combined_attention_mask.shape[1]
                        else:
                            interval = cache['interval']
                    else:
                        interval = cache['interval']
                        
                    if self.input_stitching: #先缝合再forward
                        # input_stitching为拼接后的input ids
                        input_stitching_before = cache['input_stitching_dic']['input_stitching']
                        position_ids = cache['input_stitching_dic']['position_ids']
                        combined_attention_mask = input_stitching_before.ne(self.tokenizer.pad_token_id)
                        # print("input_stitching_before.shape:{}".format(input_stitching_before.shape))
                        # # 先decode再encode
                        # input_text = self.tokenizer.decode(input_stitching_before[0],skip_special_tokens=True)
                        # print("input_text:{}".format(input_text))
                        # input_stitching = self.tokenizer.encode_plus(input_text,
                        #                                             truncation = True, 
                        #                                             return_tensors='pt', 
                        #                                             add_special_tokens=True).to(self.device).input_ids
                        # print("input_stitching.shape:{}".format(input_stitching.shape))
                        # result = torch.equal(input_stitching,input_stitching_before)
                        # print("equal or not:{}".format(result))
                        input_stitching = input_stitching_before
                        windows_key_values = None
                        del cache['past_key_values']
                        tokenized_inputs_raw = self.tokenizer.encode_plus(input, 
                                                                  truncation = True, 
                                                                  return_tensors='pt', 
                                                                  add_special_tokens=special_token).to(self.device)
                        tokenized_inputs_attention_mask = tokenized_inputs_raw.attention_mask.cuda()
                        new_position_ids = torch.arange(input_max_window_size,input_max_window_size+tokenized_inputs_raw.input_ids.shape[1]).unsqueeze(0).to(self.device)
                        # print("new_position_ids:{}".format(new_position_ids.shape))
                        # print("position_ids:{}".format(position_ids.shape))
                        input_max_window_size = input_max_window_size+tokenized_inputs_raw.input_ids.shape[1]
                        tokenized_inputs = torch.cat((input_stitching,tokenized_inputs_raw.input_ids),dim=1)
                        combined_attention_mask = torch.cat((combined_attention_mask, tokenized_inputs_attention_mask),dim=1)
                        
                        # 直接对position_ids进行操作
                        torch.set_printoptions(threshold=float('inf'))
                        position_ids = torch.cat((position_ids,new_position_ids),dim=1)
                        
                        # print("position_ids:{}".format(position_ids))
                        # print("position_ids.shape{}".format(position_ids.shape))
                        # print("tokenized_inputs.shape{}".format(tokenized_inputs.shape))
                        # print("tokenized_inputs_attention_mask.shape{}".format(combined_attention_mask.shape))
                        # assert 1==0
                        windows_key_values=None
                        context_length = tokenized_inputs.shape[1]
                        
                        # chunk_llama
                        if self.positional_sorting == "chunkllama":
                            res = self.model2.generate(input_ids=tokenized_inputs,
                                                attention_mask=combined_attention_mask,
                                                num_beams=1,
                                                do_sample=False,
                                                temperature=1.0,
                                                min_length=context_length+1,
                                                max_new_tokens=64,
                                                **kwargs)[0]
                            res = self.tokenizer.decode(res[context_length:], skip_special_tokens=True)
                            print(f"res is :{res}")
                            # assert 1==0
                            return res
                    else:
                        # print("tokenized_inputs.shape{}".format(tokenized_inputs.shape))
                        # print("tokenized_inputs_attention_mask.shape{}".format(combined_attention_mask.shape))
                        position_ids=None    
                        windows_key_values=cache['past_key_values']
                        # assert 1==0
                        
                    res = self.model.generate(input_ids=tokenized_inputs,
                                                attention_mask=combined_attention_mask,
                                                windows_key_values=windows_key_values,
                                                max_window_size=input_max_window_size,
                                                interval=interval,
                                                interval_shift=self.interval_shift,
                                                sum_windows_size=sum_windows_size,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                max_new_tokens=output_max_len, 
                                                position_dict=cache['position_dict'],
                                                num_beams=1,
                                                do_sample=False,
                                                temperature=1,
                                                min_length=context_length+1,
                                                position_ids=position_ids,
                                                labels=tokenized_inputs,
                                                **kwargs)[0]
                    # new_res = self.model(input_ids=tokenized_inputs,
                    #                  attention_mask=combined_attention_mask,
                    #                  position_ids=position_ids,
                    #                  )
                    # predict_token = torch.argmax(new_res.logits, dim=-1)
                    # print(f"predict_token: {predict_token[:,-tokenized_inputs_raw.input_ids.shape[1]:]}")
                    #logger.info(f"res.shape is :{res[context_length:].shape}")
                    torch.cuda.empty_cache()
                    if "default" not in parallel_patterns:
                        # print("res[context_length:]",res[context_length:])
                        res = self.tokenizer.decode(res[context_length:], skip_special_tokens=True)
                    else:
                        res = self.tokenizer.decode(res, skip_special_tokens=True)
                    logger.debug(f"res is: {res}")
                    # print(f"res is :{res}")
                    # assert 1==0
                    # assert 1==0
                else:
                    
                    past_key_values = cache['past_key_values']
                    # cache['sum_windows_size']  = cache['past_attention_mask'].shape[1]
                    # logger.info(f"cache['sum_windows_size']: {cache['sum_windows_size']}")
                    
                    logger.info(f"cache['sum_windows_size']: {cache['sum_windows_size']}")
                    logger.info(f"past_attention_mask.shape: {cache['past_attention_mask'].shape}")
                    logger.info(f"cache['past_key_values'][0][0].shape: {cache['past_key_values'][0][0].shape}")
                    # assert cache['sum_windows_size'] == cache['past_key_values'][0][0].shape[2]
                    assert cache['sum_windows_size'] == cache['past_attention_mask'].shape[1]
                    input_max_window_size=cache['max_raw_window_size']
                    
                    if "every_window_query_input_no_query" in parallel_patterns and "anchor" not in parallel_patterns:
                        input = "\n"
                    elif "every_window_no_query_input_query" in parallel_patterns:
                        # assert 1==0
                        input = question
                    elif "every_window_query_input_query" in parallel_patterns and "anchor" not in parallel_patterns:
                        input = question
                    elif "anchor" in parallel_patterns:
                        input = question
                    else:
                        input = "\n"
                    logger.info(f"input is: {input}")
                    #控制位置编码位置
                    if raw_position_select:
                        input_max_window_size = cache['max_raw_window_size']
                    else:
                        input_max_window_size = cache['max_window_size']
                    
                    if self.position_shift:
                        input_max_window_size = cache['max_position_idx']
                    print("input_max_window_size:{}".format(input_max_window_size))
                    
                    special_token = self.special_token
                    input = self.apply_qwen2(input)
                    tokenized_inputs = self.tokenizer.encode_plus(input, 
                                                                  truncation = True, 
                                                                  return_tensors='pt', 
                                                                  add_special_tokens=special_token)
                    tokenized_inputs_attention_mask = tokenized_inputs.attention_mask.cuda()
                    context_length = tokenized_inputs.input_ids.shape[1]
                    tokenized_inputs = tokenized_inputs.input_ids.cuda()
                    input_ids_length = tokenized_inputs.shape[1]+cache['max_window_size']
                    
                    if "default" in parallel_patterns:
                        assert len(cache['first_token']) == 1
                        # 获得第一个窗口的first_token
                        tokenized_inputs = cache['first_token'][0]
                        print("tokenized_inputs:{}".format(tokenized_inputs))
                        context_length = tokenized_inputs.shape[1]
                        tokenized_inputs = tokenized_inputs
                        tokenized_inputs_attention_mask = tokenized_inputs.ne(self.tokenizer.pad_token_id)
                        output_max_len=output_max_len-1
                    
                    #logger.info(f"A window: generate all tokens length: {tokenized_inputs.shape[1]+cache['max_window_size']}")
                    
                    # 第二次 truncation 确保添加query后仍在模型最大长度内
                    if self.NTK_aware: # 不需要截断
                        pass
                    else:
                        if input_ids_length > raw_model_max_len and "default" not in parallel_patterns:
                            logger.info("begin trucate input_ids")
                            logger.info(f"model_max_length: {raw_model_max_len}")
                            logger.info(f"input_ids_length: {input_ids_length}")
                            logger.info(f"cache['max_window_size']: {cache['max_window_size']}")
                            half = (raw_model_max_len-cache['max_window_size'])//2
                            # windows+query本身就超过了model_max_length 暂时先不处理 （可能需要考虑截断）
                            assert half > 0
                            
                            truncation_input = [self.tokenizer.decode(tokenized_inputs[i][:half], skip_special_tokens=True)+
                                  self.tokenizer.decode(tokenized_inputs[i][-half:], skip_special_tokens=True) 
                                  for i in range(len(tokenized_inputs))]
                            truncation_input = self.apply_qwen2(truncation_input[0])
                            truncation_input = [truncation_input]
                            new_tokenized_inputs = self.tokenizer(truncation_input, 
                                                                  truncation=True,
                                                                  return_tensors='pt', 
                                                                  add_special_tokens=special_token)
                            
                            tokenized_inputs_attention_mask = new_tokenized_inputs.attention_mask.cuda()
                            tokenized_inputs = new_tokenized_inputs.input_ids.cuda()
                            context_length = tokenized_inputs.shape[1]

                    logger.info(f"input tokens length: {tokenized_inputs.shape[1]}")
                    logger.info(f"A window: after truncation generate all tokens length: {tokenized_inputs.shape[1]+cache['max_window_size']}")
                    
                    sum_windows_size = cache['sum_windows_size']
                    
                    combined_attention_mask = torch.cat((cache['past_attention_mask'], tokenized_inputs_attention_mask),dim=1)
                    logger.info(f"combined_attention_mask.shape: {combined_attention_mask.shape}")
                    logger.info(f"input_max_window_size: {input_max_window_size}")
                    logger.info(f"sum_windows_size: {sum_windows_size}")
                    assert combined_attention_mask.shape[1]-sum_windows_size == tokenized_inputs_attention_mask.shape[1]
                    logger.info(f"interval: {cache['interval']}")
                    logger.info(f"interval_shift: {self.interval_shift}")
                    interval = cache['interval']
                    position_ids=None    
                    windows_key_values=past_key_values
                    # 汇总成一个大的past_key_value
                    if "list" not in parallel_patterns:
                        new_windows_key_values = HybridCache(config=self.model.config,
                                                       max_batch_size=1,
                                                       max_cache_len=cache['sum_windows_size'],
                                                       device=self.device,
                                                       dtype=self.model.dtype,)
                        new_windows_key_values.key_cache = [torch.cat([pst.key_cache[idx] for pst in past_key_values], dim=2) for idx in range(len(past_key_values[0].key_cache))]
                        new_windows_key_values.value_cache = [torch.cat([pst.value_cache[idx] for pst in past_key_values], dim=2) for idx in range(len(past_key_values[0].value_cache))]
                        self.model.label = 0
                    else:
                        new_windows_key_values = HybridCache(config=self.model.config,
                                                       max_batch_size=1,
                                                       max_cache_len=combined_attention_mask.shape[1],
                                                       device=self.device,
                                                       dtype=self.model.dtype,)
                        new_windows_key_values.key_cache = [kv_cache[0] for kv_cache in past_key_values]
                        new_windows_key_values.value_cache = [kv_cache[1] for kv_cache in past_key_values]
                        print("len(past_key_values):{}".format(len(new_windows_key_values.key_cache)))
                        print("past_key_values[0].key_cache[0].shape:{}".format(new_windows_key_values.key_cache[0].shape))
                        # print("past_key_values:{}".format(past_key_values))
                        # print("len(past_key_values[0]):{}".format(len(past_key_values[0])))
                        # assert 1==0
                        self.model.label = 0
                        
                        print("tokenized_inputs:{}".format(tokenized_inputs))
                        print("combined_attention_mask:{}".format(combined_attention_mask))
                        print("new_windows_key_values:{}".format(new_windows_key_values))
                        print("input_max_window_size:{}".format(input_max_window_size))
                        print("interval:{}".format(interval))
                        print("sum_windows_size:{}".format(sum_windows_size))
                        print("output_max_len:{}".format(output_max_len))
                       
                        # assert 1==0
                    res = self.model.generate(input_ids=tokenized_inputs,
                                                attention_mask=combined_attention_mask,
                                                windows_key_values=new_windows_key_values,
                                                max_window_size=input_max_window_size,
                                                interval=interval,
                                                interval_shift=self.interval_shift,
                                                sum_windows_size=sum_windows_size,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                max_new_tokens=output_max_len, 
                                                position_dict=cache['position_dict'],
                                                num_beams=1,
                                                do_sample=False,
                                                temperature=1,
                                                min_length=context_length+1,
                                                position_ids=position_ids,
                                                labels=tokenized_inputs,
                                                **kwargs)[0]
                    torch.cuda.empty_cache()
                    if "default" not in parallel_patterns:
                        res = self.tokenizer.decode(res[context_length:], skip_special_tokens=True)
                    else:
                        res = self.tokenizer.decode(res, skip_special_tokens=True)
                    logger.debug(f"res is: {res}")
            return res