from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from logits_processor import RestrictiveTokensLogitsProcessor
from utils import n_tokens_in_prompt
from tqdm import tqdm 
import logging
from my_utils.logger import Logger
logger = Logger()
logger.set_console_level(logging.DEBUG)

def combine_past_key_values(past_lst: List[Tuple[Tuple[torch.Tensor]]], longest_window_id: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    # We eliminate all but one bos token from windows to avoid multiple bos, which deterred our results.
    n_layers = len(past_lst[0])
    longest_window = past_lst[longest_window_id]
    all_windows_except_longest = past_lst[:longest_window_id] + past_lst[longest_window_id + 1:]
    return tuple(
        (torch.cat([longest_window[i][0]] + [c[i][0][:, :, 1:, :] for c in all_windows_except_longest], dim=2),
         torch.cat([longest_window[i][1]] + [c[i][1][:, :, 1:, :] for c in all_windows_except_longest], dim=2))
        for i in range(n_layers))

def combine_past_key_values_longbench(past_lst: List[Tuple[Tuple[torch.Tensor]]], longest_window_id: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    # We eliminate all but one bos token from windows to avoid multiple bos, which deterred our results.
    n_layers = len(past_lst[0])
    first_window = past_lst[0]
    other_windows = past_lst[1:]
    return tuple(
        (torch.cat([first_window[i][0]] + [c[i][0][:, :, 1:, :] for c in other_windows], dim=2),
         torch.cat([first_window[i][1]] + [c[i][1][:, :, 1:, :] for c in other_windows], dim=2))
        for i in range(n_layers))

def generate_pcw_position_ids(attention_mask: torch.Tensor, max_window_size: int,
                              past_key_values: Tuple[Tuple[torch.Tensor]],
                              sum_windows_size: int, windows_key_values: Tuple[Tuple[torch.Tensor]]) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    
    # contains_zero = (attention_mask == 0).any().item()
    # print(f"Attention mask contains 0: {contains_zero}")
    # torch.set_printoptions(threshold=float('inf'))
    # print("position_ids: {}".format(position_ids))
    
    # assert 1==0
    # print("position_ids shape3:{}".format(position_ids.shape))
    # print("sum_windows_size:{}".format(sum_windows_size))
    # print("max_window_size:{}".format(max_window_size))
    
    n_task_tokens = position_ids.shape[1] - sum_windows_size
    #print("n_task_tokens:{}".format(n_task_tokens))
    if n_task_tokens > 0:
       position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
    #print("torch.arange(max_window_size, max_window_size + n_task_tokens, 1):{}".format(torch.arange(max_window_size, max_window_size + n_task_tokens, 1)))
    position_ids.masked_fill_(attention_mask == 0, 1)
    # print("position_ids after 1:{}".format(position_ids))
    if past_key_values:  # i.e., first token is already generated
        # logger.info(f"position_ids shape0:{position_ids.shape}")
        position_ids = position_ids[:, -1].unsqueeze(-1)
    elif windows_key_values:  # i.e., we are in the first token generation
        #print("position_ids shape0:{}".format(position_ids.shape))
        # logger.info(f"position_ids shape1:{position_ids.shape}")
        # logger.info(f"sum_windows_size:{sum_windows_size}")
        position_ids = position_ids[:, sum_windows_size:]
    # print("position_ids after :{}".format(position_ids))
    return position_ids


class PCWModelWrapper:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 device: str,
                 context_window_size: int,
                 right_indentation: bool = False,
                 prompt_method: str = None,
                 n_windows: int=1
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_method = prompt_method
        if prompt_method == "complex_cot" or prompt_method == "complex_cot_pcw" or self.prompt_method =="complex_cot_pcw_pre_process_window_cache" or self.prompt_method == "complex_cot_pcw_multi_windows" or self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("!!!!!!!!!!!!!!!!!!!!!!!! 这里")
        #assert 1==0
        self.tokenizer.padding_side = "left"
        self.context_window_size = context_window_size
        self.device = device
        self.n_windows = n_windows
        # Left indentation is the default behavior as explained in the paper.
        self.right_indentation = right_indentation

    def _get_windows(self, texts: List[str]) -> List[Dict]:
        windows = []
        """
        texts: 传过来的不同上下文:
        
        """
        #print("self.right_indentation:{}".format(self.right_indentation))
 
        # 这两行代码的目的是在 判断右对齐选项是否启用 后，计算给定 texts 列表中每个文本的 token 数量，并获取这些 token 数量的最大值。以下是逐步解析：
        if self.right_indentation:
            #assert 1==0
            max_window_size = max(n_tokens_in_prompt(self.tokenizer, t, add_special_tokens=True) for t in texts)

        for text in texts:
            # logger.debug(f"text in _get_windows:{text}")
            # logger.debug(f"len(text):{len(text)}")
            encoded_input_window = self.tokenizer(text, return_tensors='pt').to(self.device)
            window_size = encoded_input_window['input_ids'].shape[1]
            
            if self.right_indentation:
                shift = max_window_size - window_size
                encoded_input_window["position_ids"] = encoded_input_window["attention_mask"].cumsum(-1) - 1 + shift
            # print("encoded_input_window[position_ids]:{}".format(encoded_input_window["position_ids"]))
            # assert 1==0
            with torch.no_grad():
                # 在这里完成上下文窗口的编码, prefill
                output = self.model(**encoded_input_window)
            windows.append({'text': text,
                            'encoded_input': encoded_input_window,
                            'attention_mask': encoded_input_window['attention_mask'],
                            'window_size': window_size,
                            'output': output,
                            'past': output['past_key_values']})
        return windows

    def get_contexts_cache(self, contexts: List[str]) -> Dict:
        """
            contexts： 喂入的不同上下文 windows_few_shots
            
            分段处理输入上下文（_get_windows 函数）：

            将多个上下文输入文本转化为模型的编码输入，同时生成模型的输出、历史状态（past_key_values）和注意力掩码。
            支持右对齐功能，确保对齐后输入长度一致。
            
            
            优化上下文处理和缓存（get_contexts_cache 函数）：

            从上下文窗口中提取关键信息（如 past_key_values 和 attention_mask），用于后续高效推理。
            合并上下文，避免重复的特殊标记（如 BOS token），并记录关键指标（如最大窗口大小）。
        """
        #print("len(contexts):{}".format(len(contexts)))
        # 实际上是有多少个上下文快
        windows = self._get_windows(contexts)
        logger.debug(f"len of windows in get_contexts_cache:{len(windows)}")
        
        #print("len of windows in get_contexts_cache:{}".format(len(windows)))
        windows_sizes = [window['window_size'] for window in windows]
        #print("len of windows_sizes in get_contexts_cache:{}".format(len(windows_sizes)))
        # 'sum_windows_size': sum(windows_sizes) - (len(windows) - 1)
        j = np.argmax(windows_sizes)
        #print("j:{}".format(j))
        # Windows contain bos tokens, we remove all but one to avoid multiple bos
        return {'past_key_values': combine_past_key_values([window['past'] for window in windows], j),
                'max_window_size': max(windows_sizes),
                'past_attention_mask': torch.cat(
                    [windows[j]['attention_mask']] + [window['attention_mask'][:, 1:] for window in
                                                      windows[:j] + windows[j + 1:]], dim=1),
                'sum_windows_size': sum(windows_sizes) - (len(windows) - 1)}

    def get_contexts_cache_longbench(self, contexts: List[str]) -> Dict:
        """
            contexts： 喂入的不同上下文 windows_few_shots
            
            分段处理输入上下文（_get_windows 函数）：

            将多个上下文输入文本转化为模型的编码输入，同时生成模型的输出、历史状态（past_key_values）和注意力掩码。
            支持右对齐功能，确保对齐后输入长度一致。
            
            
            优化上下文处理和缓存（get_contexts_cache 函数）：

            从上下文窗口中提取关键信息（如 past_key_values 和 attention_mask），用于后续高效推理。
            合并上下文，避免重复的特殊标记（如 BOS token），并记录关键指标（如最大窗口大小）。
        """
        #print("len(contexts):{}".format(len(contexts)))
        # 实际上是有多少个上下文快
        windows = self._get_windows(contexts)
        logger.debug(f"len of windows in get_contexts_cache:{len(windows)}")
        
        #print("len of windows in get_contexts_cache:{}".format(len(windows)))
        windows_sizes = [window['window_size'] for window in windows]
        #print("len of windows_sizes in get_contexts_cache:{}".format(len(windows_sizes)))
        # 'sum_windows_size': sum(windows_sizes) - (len(windows) - 1)
        j = np.argmax(windows_sizes)
        #print("j:{}".format(j))
        # Windows contain bos tokens, we remove all but one to avoid multiple bos
        return {'past_key_values': combine_past_key_values_longbench([window['past'] for window in windows], j),
                'max_window_size': max(windows_sizes),
                'past_attention_mask': torch.cat(
                    [windows[j]['attention_mask']] + [window['attention_mask'][:, 1:] for window in
                                                      windows[:j] + windows[j + 1:]], dim=1),
                'sum_windows_size': sum(windows_sizes) - (len(windows) - 1)}

    
    def get_contexts_cache_kv_cache_compression(self, contexts: List[str]) -> Dict:
        """
            contexts： 喂入的不同上下文 windows_few_shots
            
            分段处理输入上下文（_get_windows 函数）：

            将多个上下文输入文本转化为模型的编码输入，同时生成模型的输出、历史状态（past_key_values）和注意力掩码。
            支持右对齐功能，确保对齐后输入长度一致。
            
            
            优化上下文处理和缓存（get_contexts_cache 函数）：

            从上下文窗口中提取关键信息（如 past_key_values 和 attention_mask），用于后续高效推理。
            合并上下文，避免重复的特殊标记（如 BOS token），并记录关键指标（如最大窗口大小）。
        """
        #print("len(contexts):{}".format(len(contexts)))
        # 实际上是有多少个上下文快
        windows = self._get_windows(contexts)
        #print("len of windows in get_contexts_cache:{}".format(len(windows)))
        windows_sizes = [window['window_size'] for window in windows]
        #print("len of windows_sizes in get_contexts_cache:{}".format(len(windows_sizes)))
        # 'sum_windows_size': sum(windows_sizes) - (len(windows) - 1)
        j = np.argmax(windows_sizes)
        
        # print("windows[0][attention_mask].shape:{}".format(windows[0]['attention_mask'].shape))
        # print("windows[0]['attention_mask]",windows[0]['attention_mask'])
        # assert 1==0
        #print("j:{}".format(j))
        # Windows contain bos tokens, we remove all but one to avoid multiple bos
        return {'past_key_values': combine_past_key_values([window['past'] for window in windows], j),
                'max_window_size': max(windows_sizes),
                'past_attention_mask': torch.cat(
                    [windows[j]['attention_mask']] + [window['attention_mask'][:, 1:] for window in
                                                      windows[:j] + windows[j + 1:]], dim=1),
                'sum_windows_size': sum(windows_sizes) - (len(windows) - 1)}


        
    def split_prompts_into_windows(self, prompts, n_windows, task_text=None):
        # 确保 n_windows > 0
        if n_windows <= 0:
            raise ValueError("n_windows must be greater than 0.")
        
        # 将 prompts 划分为 n_windows 个窗口
        avg = len(prompts) / n_windows
        windows = []
        last = 0.0

        for i in range(1, n_windows + 1):
            next_idx = round(i * avg)
            if task_text != None:
                #print("len of prompts[int(last):int(next_idx)]:{}".format(len(prompts[int(last):int(next_idx)])))
                windows.append("\n\n".join(prompts[int(last):int(next_idx)]) + "\n\n" + task_text)
            else:
                windows.append("\n\n".join(prompts[int(last):int(next_idx)]))
            last = next_idx

        return windows
    
    def pcw_generate(self,
                     task_text: Optional[str] = None,
                     contexts: Optional[List[str]] = None,
                     contexts_cache: Optional[Dict] = None,
                     restrictive_logit_preprocessor: Optional[RestrictiveTokensLogitsProcessor] = None,
                     few_shots_prompts: Optional[str] = None,
                     **kwargs
                     ) -> str:
        """Note: Batching is not supported by PCW at the moment. """
        with torch.no_grad():
            if self.prompt_method == "other":

                assert (contexts is None) != (
                        contexts_cache is None), "pcw_generate should work with contexts or cache, not with both!"
        
                # 在contexts这里得到所有的上下文
                cache = contexts_cache or self.get_contexts_cache(contexts)
                """
                                            windows_key_values=cache['past_key_values'],
                                            max_window_size=cache['max_window_size'],
                                            sum_windows_size=cache['sum_windows_size'],
                """
                encoded_task_text = self.tokenizer(task_text, add_special_tokens=False, return_tensors='pt').to(self.device)
                if restrictive_logit_preprocessor:
                    restrictive_logit_preprocessor.update_new_prompt_length_to_skip(encoded_task_text['input_ids'].shape[1])
                    kwargs['logits_processor'] = [restrictive_logit_preprocessor]
                combined_attention_mask = torch.cat((cache['past_attention_mask'], encoded_task_text['attention_mask']),
                                            dim=1).to(self.device)
                res = self.model.generate(input_ids=encoded_task_text['input_ids'],
                                        attention_mask=combined_attention_mask,
                                        windows_key_values=cache['past_key_values'],
                                        max_window_size=cache['max_window_size'],
                                        sum_windows_size=cache['sum_windows_size'],
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        **kwargs)[0]
            elif self.prompt_method == "complex_cot_pcw":
                
         
                # cache = self.get_contexts_cache([few_shots_prompts])
                # tokenized_inputs = self.tokenizer.encode_plus(task_text, truncation = True, return_tensors='pt', add_special_tokens=False)
                # tokenized_inputs_attention_mask = tokenized_inputs.attention_mask.cuda()
                # #print("tokenized_inputs_attention_mask shape:{}".format(tokenized_inputs_attention_mask.shape))
                # combined_attention_mask = torch.cat((cache['past_attention_mask'], tokenized_inputs_attention_mask),dim=1)#tokenized_inputs.attention_mask.cuda() #torch.cat((cache['past_attention_mask'], tokenized_inputs.attention_mask.cuda()),dim=1)#.to(self.device)
                # #print("combined_attention_mask shape:{}".format(combined_attention_mask.shape))
                # res = self.model.generate(input_ids=tokenized_inputs.input_ids.cuda(),
                #                         attention_mask=combined_attention_mask,
                #                         windows_key_values=cache['past_key_values'],
                #                         max_window_size=cache['max_window_size'],
                #                         sum_windows_size=cache['sum_windows_size'],
                #                         eos_token_id=self.tokenizer.eos_token_id,
                #                         pad_token_id=self.tokenizer.pad_token_id,
                #                         **kwargs)[0]
                #print(r"task_text:{}".format(task_text))
                #assert 1==0
                # prompts = few_shots_prompts.split("\n\n")
                # print("prompts:{}".format(prompts))
                # assert 1==0
                input = few_shots_prompts + "\n\n" + task_text
                cache = self.get_contexts_cache([input])
                tokenized_inputs = self.tokenizer.encode_plus("\nA: Let's think step by step", truncation = True, return_tensors='pt', add_special_tokens=False)
                tokenized_inputs_attention_mask = tokenized_inputs.attention_mask.cuda()
                #print("tokenized_inputs_attention_mask shape:{}".format(tokenized_inputs_attention_mask.shape))
                combined_attention_mask = torch.cat((cache['past_attention_mask'], tokenized_inputs_attention_mask),dim=1)#tokenized_inputs.attention_mask.cuda() #torch.cat((cache['past_attention_mask'], tokenized_inputs.attention_mask.cuda()),dim=1)#.to(self.device)
                #print("combined_attention_mask shape:{}".format(combined_attention_mask.shape))
                res = self.model.generate(input_ids=tokenized_inputs.input_ids.cuda(),
                                        attention_mask=combined_attention_mask,
                                        windows_key_values=cache['past_key_values'],
                                        max_window_size=cache['max_window_size'],
                                        sum_windows_size=cache['sum_windows_size'],
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        **kwargs)[0]
                
                
                # input = few_shots_prompts + "\n\n" +task_text
                # tokenized_inputs = self.tokenizer.encode_plus(input, truncation = True, return_tensors='pt', add_special_tokens=False)
                # tokenized_inputs_attention_mask = tokenized_inputs.attention_mask.cuda()
 
                # res = self.model.generate(input_ids=tokenized_inputs.input_ids.cuda(),
                #                         attention_mask=tokenized_inputs_attention_mask,
   
                #                         eos_token_id=self.tokenizer.eos_token_id,
                #                         pad_token_id=self.tokenizer.pad_token_id,
                #                         **kwargs)[0]
            elif self.prompt_method == "complex_cot_pcw_multi_windows":   # 进到这里, 申江涵
                

                prompts = few_shots_prompts.split("\n\n")
                #print("len(prompts):{}".format(len(prompts)))
                # assert 1==0
                prompts_avg = self.split_prompts_into_windows(prompts, self.n_windows ,task_text=task_text)
                #print("len(prompts_avg):{}".format(len(prompts_avg)))
                #print("len(prompts_avg):{}".format(len(prompts_avg)))
#                assert 1==0
                # print("1111111111111111")
                cache = self.get_contexts_cache(prompts_avg)
                # print("2222222222222222")
                #assert 1==0
                # task_text1 + quesiton==query (prefill  kv cache pruning   recent token = (1.input 2.query))  ->  input(generate) = "\nA: Let's think step by step"#task_text + "\nA: Let's think step by step"#"\nA: Let's think step by step" #task_text + "\nA: Let's think step by step\n"
                # task_text2 + question==query
                # task_text3 + quesiton==query
                
                
                input = "\nA: Let's think step by step"
                
                tokenized_inputs = self.tokenizer.encode_plus(input, truncation = True, return_tensors='pt', add_special_tokens=False)
                tokenized_inputs_attention_mask = tokenized_inputs.attention_mask.cuda()
                #print("tokenized_inputs_attention_mask shape:{}".format(tokenized_inputs_attention_mask.shape))
                combined_attention_mask = torch.cat((cache['past_attention_mask'], tokenized_inputs_attention_mask),dim=1)#tokenized_inputs.attention_mask.cuda() #torch.cat((cache['past_attention_mask'], tokenized_inputs.attention_mask.cuda()),dim=1)#.to(self.device)
                print("combined_attention_mask shape:{}".format(combined_attention_mask.shape))
                # assert 1==0

                res = self.model.generate(input_ids=tokenized_inputs.input_ids.cuda(),
                                        attention_mask=combined_attention_mask,
                                        windows_key_values=cache['past_key_values'],
                                        max_window_size=cache['max_window_size'],
                                        sum_windows_size=cache['sum_windows_size'],
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        **kwargs)[0]
                
            elif self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":   # 进到这里, 申江涵
                

                prompts = few_shots_prompts.split("\n\n")
                #print("len(prompts):{}".format(len(prompts)))
                # assert 1==0
                prompts_avg = self.split_prompts_into_windows(prompts, self.n_windows ,task_text=task_text)
                #print("len(prompts_avg):{}".format(len(prompts_avg)))
                #print("len(prompts_avg):{}".format(len(prompts_avg)))
#                assert 1==0
                # print("1111111111111111")
                cache = self.get_contexts_cache_kv_cache_compression(prompts_avg)
                # 512 + 511*(n_windows-1)
                # capacity = 512
                # cache['past_attention_mask'] = cache['past_attention_mask'][:, :capacity+(capacity-1)*(self.n_windows-1)]
                print("2222222222222222")
                #assert 1==0
                # task_text1 + quesiton==query (prefill  kv cache pruning   recent token = (1.input 2.query))  ->  input(generate) = "\nA: Let's think step by step"#task_text + "\nA: Let's think step by step"#"\nA: Let's think step by step" #task_text + "\nA: Let's think step by step\n"
                # task_text2 + question==query
                # task_text3 + quesiton==query
                
                
                input = "\nA: Let's think step by step"
                
                tokenized_inputs = self.tokenizer.encode_plus(input, truncation = True, return_tensors='pt', add_special_tokens=False)
                tokenized_inputs_attention_mask = tokenized_inputs.attention_mask.cuda()
                #print("tokenized_inputs_attention_mask shape:{}".format(tokenized_inputs_attention_mask.shape))
                
                combined_attention_mask = torch.cat((cache['past_attention_mask'], tokenized_inputs_attention_mask),dim=1)#tokenized_inputs.attention_mask.cuda() #torch.cat((cache['past_attention_mask'], tokenized_inputs.attention_mask.cuda()),dim=1)#.to(self.device)
                # print("combined_attention_mask shape:{}".format(combined_attention_mask.shape))
                # print("cache['past_attention_mask'].shape:{}".format(cache['past_attention_mask'].shape))
                # print("tokenized_inputs_attention_mask.shape:{}".format(tokenized_inputs_attention_mask.shape))
                # assert 1==0
                
                # 并行上下文窗口+input
                # assert 1==0

                res = self.model.generate(input_ids=tokenized_inputs.input_ids.cuda(),
                                        attention_mask=combined_attention_mask,
                                        windows_key_values=cache['past_key_values'],
                                        max_window_size=cache['max_window_size'],
                                        sum_windows_size=cache['sum_windows_size'],
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        **kwargs)[0]
          
                

            elif self.prompt_method == "complex_cot_pcw_pre_process_window_cache":
#                assert 1==0
                cache = contexts_cache
                #print("cache 6:{}".format(cache))
                tokenized_inputs = self.tokenizer.encode_plus(task_text, truncation = True, return_tensors='pt', add_special_tokens=False)
                tokenized_inputs_attention_mask = tokenized_inputs.attention_mask.cuda()
                combined_attention_mask = torch.cat((cache['past_attention_mask'].cuda(), tokenized_inputs_attention_mask),dim=1)#tokenized_inputs.attention_mask.cuda() #torch.cat((cache['past_attention_mask'], tokenized_inputs.attention_mask.cuda()),dim=1)#.to(self.device)
                res = self.model.generate(input_ids=tokenized_inputs['input_ids'].cuda(),
                                        attention_mask=combined_attention_mask,
                                        windows_key_values=cache['past_key_values'],
                                        max_window_size=cache['max_window_size'],
                                        sum_windows_size=cache['sum_windows_size'],
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        **kwargs)[0]
            else:
                input = few_shots_prompts + "\n\n" +task_text
                tokenized_inputs = self.tokenizer.encode_plus(input, truncation = True, return_tensors='pt', add_special_tokens=False)
                res = self.model.generate(input_ids=tokenized_inputs.input_ids.cuda(),
                                        attention_mask=tokenized_inputs.attention_mask.cuda(),
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        **kwargs)[0]
      
        
        if res[-1] == self.tokenizer.eos_token_id or self.prompt_method == "complex_cot" or self.prompt_method == "complex_cot_pcw" or self.prompt_method =="complex_cot_pcw_pre_process_window_cache" or self.prompt_method == "complex_cot_pcw_multi_windows" or self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
            prompt_len = int(tokenized_inputs.attention_mask.shape[1])
            # generated:["Janet eats 3 eggs per day, so she has 16 - 3 = <<16-3=13>>13 eggs left for sale.\nShe sells each egg for $2, so she makes $2 x 13 = $<<2*13=26>>26 per day from selling eggs at the farmers' market.\nShe also bakes 4 muffins per day and sells each for $4, so she makes $4 x 4 = $<<4*4=16>>16 per day from muffins.\nIn total, Janet makes $26 + $16 = $<<26+16=42>>42 per day at the farmers' market.\nThe answer is 42</s>"]
            generated = [self.tokenizer.decode(res[prompt_len:])]
            #print("generated1:{}".format(generated))
            generated = [g.strip(self.tokenizer.pad_token).strip() for g in generated]
            
            return generated[0]
        else:
        #res = res[:-1] if res[-1] == self.tokenizer.eos_token_id else res
            return self.tokenizer.decode(res[encoded_task_text['input_ids'].shape[1]:])

    def pcw_generate_longbench(self,
                               per_windows_prompt: List[str],
                               output_max_len: int,
                               parallel_patterns:str,
                               question="",
                               **kwargs,
                               ):
        if self.prompt_method=="complex_cot_pcw_multi_windows":
            # parallel windows
            cache = self.get_contexts_cache_longbench(per_windows_prompt)
            if parallel_patterns == "every_window_query_input_no_query":
                input = "\n"
            elif parallel_patterns == "every_window_no_query_input_query":
                input = question
            elif parallel_patterns == "every_window_query_input_query":
                input = question
            else:
                input = "\n"
            tokenized_inputs = self.tokenizer.encode_plus(input, truncation = True, return_tensors='pt', add_special_tokens=False)
            tokenized_inputs_attention_mask = tokenized_inputs.attention_mask.cuda()
            # logger.info(f"tokenized_inputs_attention_mask.shape is :{tokenized_inputs_attention_mask.shape}")
            # logger.info(f"cache['past_attention_mask'].shape is :{cache['past_attention_mask'].shape}")
            combined_attention_mask = torch.cat((cache['past_attention_mask'], tokenized_inputs_attention_mask),dim=1)
            context_length = tokenized_inputs.input_ids.shape[1]
            res = self.model.generate(input_ids=tokenized_inputs.input_ids.cuda(),
                                        attention_mask=combined_attention_mask,
                                        windows_key_values=cache['past_key_values'],
                                        max_window_size=cache['max_window_size'],
                                        sum_windows_size=cache['sum_windows_size'],
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        max_new_tokens=output_max_len,
                                        num_beams=1,
                                        do_sample=False,
                                        temperature=1.0,
                                        min_length=context_length+1,
                                        **kwargs)[0]
            
            logger.info(f"res.shape is :{res[context_length:].shape}")
            res = self.tokenizer.decode(res[context_length:], skip_special_tokens=True)
        return res
        