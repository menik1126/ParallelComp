from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import random
import torch
from transformers import models
from transformers import PreTrainedTokenizerBase, PreTrainedModel
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

def combine_past_key_values_longbench(
                                      past_lst: List[Tuple[Tuple[torch.Tensor]]], 
                                      query_len, token_prompt_size: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    n_layers = len(past_lst[0])
    first_window = past_lst[0]
    if first_window == 0:
        query_len = [-first_window[0][0].shape[2]]*len(query_len)
    first_query_len = query_len[0]
    other_windows = past_lst[1:]
    # print("ok")
    with torch.no_grad():
        if isinstance(first_window[0][0],list): # mulit head
            n_groups = len(first_window[0][0])
            combine_past_key_values = tuple(
                tuple(
                    [
                        torch.cat([first_window[i][j][0][:,:,:-first_query_len,:]] + [c[i][j][0][:, :, 1+token_prompt_size:-query_len[index+1], :]  for index,c in enumerate(other_windows)], dim=2),
                        torch.cat([first_window[i][j][1][:,:,:-first_query_len,:]] + [c[i][j][1][:, :, 1+token_prompt_size:-query_len[index+1], :]  for index,c in enumerate(other_windows)], dim=2)
                    ]
                    for j in range(n_groups)
                )
                for i in range(n_layers)
            )
        else:                    
            combine_past_key_values = tuple(
                (
                torch.cat([first_window[i][0][:,:,:-first_query_len,:]] + [c[i][0][:, :, 1+token_prompt_size:-query_len[index+1], :]  for index,c in enumerate(other_windows) ], dim=2),
                torch.cat([first_window[i][1][:,:,:-first_query_len,:]] + [c[i][1][:, :, 1+token_prompt_size:-query_len[index+1], :]  for index,c in enumerate(other_windows)], dim=2)
                )
                for i in range(n_layers)
            )   
    del past_lst
    torch.cuda.empty_cache()
    return combine_past_key_values

def combine_past_key_values_longbench_batches(
                                      past_lst: List[Tuple[Tuple[torch.Tensor]]], 
                                      select_index: List[torch.Tensor],
                                      padding_len: List[Tuple[Tuple[torch.Tensor]]],
                                      query_len, token_prompt_size: int) -> \
    Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    n_layers = len(past_lst)
    first_window = past_lst[0]
    if first_window == 0:
        query_len = [-first_window[0][0].shape[2]]*len(query_len)
    for pad in padding_len[1:]:
        pad = pad + 1+token_prompt_size
    combine_past_key_values = tuple(
        (
        torch.cat([ past_lst[i][0][select_window_index:select_window_index+1,:,padding_index_len:-query_len[index],:] for index,(select_window_index,padding_index_len) in enumerate(zip(select_index,padding_len))], dim=2),
        torch.cat([ past_lst[i][1][select_window_index:select_window_index+1,:,padding_index_len:-query_len[index],:] for index,(select_window_index,padding_index_len) in enumerate(zip(select_index,padding_len))], dim=2),
        )
        for i in range(n_layers)
    )      
    return combine_past_key_values
        
def generate_pcw_position_ids(attention_mask: torch.Tensor, max_window_size: int,
                              past_key_values: Tuple[Tuple[torch.Tensor]],
                              sum_windows_size: int, windows_key_values: Tuple[Tuple[torch.Tensor]],
                              interval: float, 
                              ) -> torch.Tensor:
    position_ids = torch.arange(0, (attention_mask.shape[1]+interval)*interval, interval)[:attention_mask.shape[1]].unsqueeze(0).to(attention_mask.device)
    n_task_tokens = position_ids.shape[1] - sum_windows_size
    if n_task_tokens > 0:
        position_ids[0, -n_task_tokens:] = torch.arange(max_window_size*interval, (max_window_size + n_task_tokens+1)*interval, interval)[:n_task_tokens]
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past_key_values: # generate
        position_ids = position_ids[:, -1].unsqueeze(-1)
    elif windows_key_values:  # second prefill
        position_ids = position_ids[:, sum_windows_size:]    
    return position_ids

class PCWModelWrapperBatches:
    def __init__(self,
                 model: PreTrainedModel,tokenizer: PreTrainedTokenizerBase,device: str,context_window_size: int,
                 # base parameters
                 model_name="",
                 parallel_pattern:str=None,
                 raw_model_max_len:int = 3950, special_token:bool=True,
                 context_prompt:dict=None,
                 # window parameters
                 topk_windows:int=1,
                 n_windows: int=1,query_rank:bool=False,
                 query_recent_tokens:int=0,
                 # kv cache eviction parameters
                 capacity:int=0,kv_cache_eviction:bool=False,
                 stage_eviction:bool=False,
                 # attention calibration
                 calibration_stage:str=None,calibration_mode:int=0,
                 ):
        # attntion calibration         
        self.calibration_stage = calibration_stage
        self.calibration_mode = calibration_mode
        self.stream = torch.cuda.Stream()
        # base parameters
        self.model = model
        self.raw_model_max_len = raw_model_max_len
        self.tokenizer = tokenizer
        self.parallel_pattern= parallel_pattern
        self.device = device
        self.special_token = special_token
        self.context_prompt=context_prompt
        # kv cache eviction
        self.kv_cache_eviction = kv_cache_eviction
        self.capacity = capacity
        self.stage_eviction = stage_eviction
        # dynamic window
        self.topk_windows = topk_windows
        self.query_rank = query_rank
        self.query_recent_tokens = query_recent_tokens
        self.model_name= model_name

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
        
        answer_lens = (shift_labels != pad_token_id).sum(-1)
        # logger.info("answer_lens: {}".format(answer_lens))
        # logger.info("loss.shape 0: {}".format(loss.shape))
        all_loss = loss.sum(-1) / (answer_lens - 1)
        
        loss = all_loss.cpu().detach().numpy().tolist()
        return loss
    
    def nll_loss_query(self, logits, tokenized_example, query_len):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits_query = shift_logits[:,-query_len:,:] 
        shift_labels = (tokenized_example['input_ids'][..., 1:]).contiguous()
        shift_labels_query = shift_labels[:,-query_len:] 
        
        pad_token_id = self.tokenizer.pad_token_id
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
        
        
        loss = loss_fct(shift_logits_query.view(-1, shift_logits_query.size(-1)),
                    shift_labels_query.view(-1)).view(shift_labels_query.size())
        
        answer_lens = (shift_labels_query != pad_token_id).sum(-1)
        all_loss = loss.sum(-1) / (answer_lens - 1)
        
        loss = all_loss.cpu().detach().numpy().tolist()
        return loss
    
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
        # elif "llama-3" in self.model_name.lower():
        #     # truncation_prompts = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{truncation_prompts}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        #     pass
        return truncation_prompts
    
    def get_window(self, text , raw_location:int,
                    raw_model_max_len:int, context_max_len:int, 
                   recent_token:int,first_max_len:int
                   ):
        if raw_location == 0:
            for i in range(len(self.model.model.layers)):
                self.model.model.layers[i].self_attn.token_prompt_size = 0
        else:
            for i in range(len(self.model.model.layers)):
                self.model.model.layers[i].self_attn.token_prompt_size = self.token_prompt_size + 1
        
        if "parallel_comp" in self.parallel_pattern:
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
            
            encoded_input_window = {
                "input_ids":
                    torch.cat([encoded_input_window_context["input_ids"],encoded_input_window_query["input_ids"]],dim=1),
                "attention_mask":
                    torch.cat([encoded_input_window_context["attention_mask"],encoded_input_window_query["attention_mask"]],dim=1),
                }
        else:
            text = self.apply_qwen2(text)
            encoded_input_window = self.tokenizer(text,
                                          padding="longest", 
                                          return_tensors="pt", 
                                          add_special_tokens=True,
                                          ).to(self.device)
        if "parallel_comp" in self.parallel_pattern:
            if len_context > first_max_len :
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
                len_query = min(raw_model_max_len-first_max_len-1,len_query)
                assert len_query <= raw_model_max_len-context_max_len 
            elif len_context+len_query > first_max_len :
                len_query = min(raw_model_max_len-len_context-1,len_query)
                
            encoded_input_window = {
                "input_ids":
                    torch.cat([encoded_input_window_context["input_ids"],encoded_input_window_query["input_ids"][:,-len_query:]],dim=1),
                "attention_mask":
                    torch.cat([encoded_input_window_context["attention_mask"],encoded_input_window_query["attention_mask"][:,-len_query:]],dim=1),
            }
        
        if "parallel_comp" in self.parallel_pattern : 
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
        else:
            query_len = 0
        window_size = encoded_input_window['input_ids'].shape[1]
        max_position_idx = encoded_input_window["attention_mask"].shape[1]
        interval = 1
        return {'text': text,
                'encoded_input': encoded_input_window,
                'attention_mask': encoded_input_window['attention_mask'],
                'window_size': window_size,
                'query_len': query_len,
                'max_position_idx': max_position_idx,
                'interval':interval
                }
    
    def _get_windows_longbench(self, texts: List[str],context_max_len:int,
                               raw_model_max_len:int,
                               recent_token:int
                               ) -> List[Dict]:
        windows = []
        first_max_len = context_max_len+(raw_model_max_len-context_max_len)//2
        if self.topk_windows < 0:
            size = -self.topk_windows
        else:
            size = self.topk_windows
        pq = priorityqueue.FixedSizePriorityQueue(size=size,key="loss")
        if "parallel_comp" in self.parallel_pattern:
            min_context_len = min(n_tokens_in_prompt(self.tokenizer, t['context'], add_special_tokens=False) for t in texts)
            self.capacity = min(self.capacity,min_context_len)
        
        input_ids_all = []
        attention_mask_all = []
        if "batches" in self.parallel_pattern:
            my_windows = []
            for raw_location,text in enumerate(texts):
                window = self.get_window(text,raw_location,raw_model_max_len,context_max_len,recent_token,first_max_len)
                my_windows.append(window)
        my_dtype = my_windows[0]['encoded_input']["input_ids"].dtype 
        mask_dtype = my_windows[0]['encoded_input']["attention_mask"].dtype  
        max_len = max([window['encoded_input']["input_ids"].shape[1] for window in my_windows])
        padding_len = [max_len-window['encoded_input']["input_ids"].shape[1] for window in my_windows]
        for i in range(len(my_windows)):
            my_value = self.tokenizer.pad_token_id
            my_windows[i]['encoded_input']["input_ids"] = torch.cat([torch.full((1, padding_len[i]), fill_value=my_value, dtype=my_dtype).to(self.device),
                                                                     my_windows[i]['encoded_input']["input_ids"]
                                                                    ],dim=1)
            my_windows[i]['encoded_input']["attention_mask"] = torch.cat([torch.zeros(1,padding_len[i],dtype=mask_dtype).to(self.device),
                                                                          my_windows[i]['encoded_input']["attention_mask"]
                                                                    ],dim=1)
            input_ids_all.append(my_windows[i]['encoded_input']["input_ids"])
            attention_mask_all.append(my_windows[i]['encoded_input']["attention_mask"])
        encoded_input_window = {}
        input_ids_all = torch.cat(input_ids_all,dim=0)
        attention_mask_all = torch.cat(attention_mask_all,dim=0)
        encoded_input_window["input_ids"] = input_ids_all
        encoded_input_window["attention_mask"] = attention_mask_all
        with torch.no_grad():
            output = self.model(**encoded_input_window)
        
        for i in range(len(my_windows)):
            new_output = {}
            new_output['logits'] = output.logits[i:i+1][:,padding_len[i]:,:] 
            my_windows[i]['output'] = new_output
        
        for raw_location,text in enumerate(texts):
            if "batches" in self.parallel_pattern:
                window = my_windows[raw_location]
                pass
            else:
                window = self.get_window(text,raw_location,raw_model_max_len,context_max_len,recent_token,first_max_len)
            
            logits = window['output']['logits']
            tokenized_example = window['encoded_input']
            eviction_lens = []
            for layer in self.model.model.layers:
                eviction_lens.append(layer.self_attn.eviction_len)
            window["eviction_len"] = eviction_lens
            
            if self.query_rank:
                if self.query_recent_tokens>0:
                    loss = self.nll_loss_query(logits, tokenized_example, min(self.query_recent_tokens,window['query_len']))[0]
                else:
                    loss = self.nll_loss_query(logits, tokenized_example, window['query_len'])[0]
            else:
                loss = self.nll_loss_all(logits, tokenized_example['input_ids'])[0]
            if self.topk_windows < 0: 
                loss = -loss
            window['raw_location'] = raw_location
            if "shuffle" in self.parallel_pattern:
                window['loss'] = random.uniform(0,3)
            else:
                window['loss'] = loss
            pq.add(window)
            windows = pq.get_elements_key("raw_location")
            
        select_index = [window['raw_location'] for window in windows]
          
        logger.info(f"raw_location: {[window['raw_location'] for window in windows]}") 
        if "default" in self.parallel_pattern:
            for window in windows:
                window["eviction_len"] = [0]*32
            padding_len = [0]
            select_index = [0]
        else :
            padding_len = [padding_len[idx_item] for idx_item in select_index]
        return windows,select_index,output.past_key_values,padding_len
     
    def get_contexts_cache_longbench(self, contexts: List[str],context_max_len:int,
                                     raw_model_max_len:int,
                                     recent_token:int
                                     ) -> Dict:
        token_prompt_size = 0
        for i in range(len(self.model.model.layers)):
            self.model.model.layers[i].self_attn.token_prompt_size = token_prompt_size 
            self.token_prompt_size = token_prompt_size 
        windows,select_idx,batch_past_key_values,padding_len = \
            self._get_windows_longbench(contexts,context_max_len,raw_model_max_len,recent_token)
        logger.debug(f"len of windows in get_contexts_cache:{len(windows)}")
        if self.kv_cache_eviction:
            logger.debug(f"self.capacity: {self.capacity}")
            windows_sizes = [min(self.capacity,window['window_size']-window['query_len']) for window in windows]
            capacity_add_query = [min(self.capacity,window['window_size']-window['query_len'])+window['query_len'] for window in windows]
            max_window_size = max([window['window_size']-window['query_len'] for window in windows])
            if "parallel_comp" not in self.parallel_pattern: # 不裁剪query
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
            if "parallel_comp" not in self.parallel_pattern: # 不裁剪query
                for window in windows:
                    window['query_len'] = -window['encoded_input'].input_ids.shape[1]
            past_attention_mask = torch.cat([windows[0]['attention_mask'][:,:-windows[0]['query_len']]] + 
                        [window['attention_mask'][:, 1+token_prompt_size:-window['query_len']] 
                         for window in windows[1:]], dim=1)

        raw_windows_sizes = [window['window_size'] for window in windows]
        predict_token = [torch.argmax(window['output']['logits'], dim=-1) for window in windows]
        
        max_raw_window_size = max(raw_windows_sizes)
        raw_sum_windows_size = sum(raw_windows_sizes) - (len(windows) - 1)
        sum_windows_size = sum(windows_sizes) - (len(windows) - 1)
        max_window = max(windows, key=lambda window: window['max_position_idx'])
        
        interval = max([window['interval'] for window in windows])
        past_key_values = combine_past_key_values_longbench_batches(batch_past_key_values,
                                                                    query_len=[window['query_len'] for window in windows],
                                                                    select_index=select_idx,padding_len=padding_len,
                                                                    token_prompt_size=token_prompt_size,
                                                                    )
     
        
        return {'past_key_values': past_key_values,
                'max_window_size': max_window_size,
                'max_raw_window_size': max_raw_window_size,
                'past_attention_mask': past_attention_mask,
                'sum_windows_size': sum_windows_size,
                'raw_sum_windows_size': raw_sum_windows_size,
                'first_token': [predict_token[i][:,-1:] for i in range(len(windows))],
                'max_position_idx': max_window['max_position_idx'],
                'interval': interval,
                }

    def pcw_generate_longbench(self,
                               per_windows_prompt: List[str],
                               output_max_len: int,
                               parallel_patterns:str,
                               question="",
                               context_max_len=3600,
                               raw_model_max_len=3950,
                               recent_token:int=8,
                               **kwargs,
                               ):
        with torch.inference_mode():
            cache = self.get_contexts_cache_longbench(per_windows_prompt,context_max_len=context_max_len,
                                                      raw_model_max_len=raw_model_max_len,recent_token=recent_token)
        
            past_key_values = cache['past_key_values']
            print(type(past_key_values[0][0]))
            if isinstance(past_key_values[0][0], list):
                past_key_values = cache['past_key_values']
                logger.info(f"cache['sum_windows_size']: {cache['sum_windows_size']}")
                logger.info(f"past_attention_mask.shape: {cache['past_attention_mask'].shape}")
                logger.info(f"cache['past_key_values'][0][0][0].shape: {cache['past_key_values'][0][0][0].shape}")
                # assert cache['sum_windows_size'] == cache['past_key_values'][0][0].shape[2]
                assert cache['sum_windows_size'] == cache['past_attention_mask'].shape[1]
                
                if "parallel_comp" in parallel_patterns:
                    input = question
                else:
                    input = "\n"
                logger.info(f"input is: {input}")
                input_max_window_size = cache['max_window_size']
                assert 1==0
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
                
                interval = cache['interval']
                
                position_ids=None    
                windows_key_values=cache['past_key_values']
                res = self.model.generate(input_ids=tokenized_inputs,
                                            attention_mask=combined_attention_mask,
                                            windows_key_values=windows_key_values,
                                            max_window_size=input_max_window_size,
                                            interval=interval,
                                            sum_windows_size=sum_windows_size,
                                            eos_token_id=self.tokenizer.eos_token_id,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            max_new_tokens=output_max_len, 
                                            num_beams=1,
                                            do_sample=False,
                                            temperature=1,
                                            min_length=context_length+1,
                                            position_ids=position_ids,
                                            labels=tokenized_inputs,
                                            **kwargs)[0]
                torch.cuda.empty_cache()
                if "default" not in parallel_patterns:
                    # print("res[context_length:]",res[context_length:])
                    res = self.tokenizer.decode(res[context_length:], skip_special_tokens=True)
                else:
                    res = self.tokenizer.decode(res, skip_special_tokens=True)
                logger.debug(f"res is: {res}")
        
                return res
            else:
                logger.info(f"cache['sum_windows_size']: {cache['sum_windows_size']}")
                logger.info(f"past_attention_mask.shape: {cache['past_attention_mask'].shape}")
                logger.info(f"cache['past_key_values'][0][0].shape: {cache['past_key_values'][0][0].shape}")
                assert cache['sum_windows_size'] == cache['past_attention_mask'].shape[1]
                
                if "parallel_comp" in parallel_patterns:
                    input = question
                else:
                    input = "\n"
                logger.info(f"input is: {input}")
                #控制位置编码位置
                
                input_max_window_size = cache['max_window_size']
                
                print("input_max_window_size:{}".format(input_max_window_size))
                
                special_token = self.special_token
                
                input = self.apply_qwen2(input)
                tokenized_inputs = self.tokenizer.encode_plus(input, 
                                                              truncation = True, 
                                                              return_tensors='pt',)
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
                
              
            
                interval = cache['interval']
                    
                position_ids=None    
                windows_key_values=cache['past_key_values']
                
            res = self.model.generate(input_ids=tokenized_inputs,
                                        attention_mask=combined_attention_mask,
                                        windows_key_values=windows_key_values,
                                        max_window_size=input_max_window_size,
                                        interval=interval,
                                        sum_windows_size=sum_windows_size,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        max_new_tokens=output_max_len, 
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