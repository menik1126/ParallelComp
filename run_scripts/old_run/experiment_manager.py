import logging
from pathlib import Path
import random
from typing import List, Dict
import math
import re
from accelerate import Accelerator
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
import json
from constants import TEXT_BETWEEN_SHOTS, N_TOKENS, PROMPTS, TEXT_BETWEEN_SHOTS_CoT
from datasets_loader import LABEL_TOKENS
from pcw_wrapper import PCWModelWrapper
from logits_processor import RestrictiveTokensLogitsProcessor
from utils import n_tokens_in_prompt, encode_labels, encode_stop_seq


from accelerate.utils import gather_object
import torch
_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')
from my_utils.logger import Logger
logger = Logger()
logger.set_console_level(logging.DEBUG)
import os



class ExperimentManager_longbench:
    def __init__(self, data_file: str, model: PCWModelWrapper, random_seed: int = 42, n_windows: int = 1, 
                 # kv cache parameters
                 kv_cache_eviction:bool=False, kv_cache_dynamic:bool=False,recent_token:int=8,
                 raw_position_select:bool=True,capacity:int=512,
                 stage_eviction:bool=False,special_token:bool=True,
                 ## calibration attn
                 calibration_mode:int=0,calibration_stage:str=None,
                 # window parameters
                 rank_windows:bool=False,rank_cache:bool=False,topk_windows:int=1,
                 dynamic_window:bool=False,window_ascend:bool=False,decomposition_factor:int=1,
                 reduce_length:bool=False,reduce_factor:int=1,
                 query_rank:bool=False,query_recent_tokens:int=0,
                 get_recent_attn:bool=False,recent_top_num:int=0,
                 key_no_rope:bool=False,delete_context_prompt:bool=False,
                 # base parameters
                 attn_avg:bool=False,
                 attn_implementation:str=None,prompt_method:str=None,model_class:str=None,parallel_pattern:str=None,
                 model_name:str=None,dataset:str=None,model2prompt=None,templates:Dict=None,
                 context_max_len: int = 1024, raw_model_max_len:int = 1024,
                 accelerator: Accelerator=None, Truncation_Method:str=None,
                 data_name:str="longbench",
                 # position shift
                 position_shift:bool=False,shift_factor:int=0,interval_shift:int=0,
                 # positional_sorting
                 positional_sorting:str=None,rank_ascend:bool=False,
                 # NTK
                 NTK_aware:bool=False,
                 ):   
        # calibration attn
        self.calibration_mode = calibration_mode
        self.calibration_stage = calibration_stage
        # window parameters
        self.dynamic_window = dynamic_window
        self.rank_windows=rank_windows
        self.rank_cache=rank_cache
        self.topk_windows=topk_windows
        self.window_ascend = window_ascend
        self.recent_token = recent_token
        self.decomposition_factor = decomposition_factor
        
        self.reduce_length = reduce_length
        self.reduce_factor = reduce_factor
        
        self.get_recent_attn = get_recent_attn
        self.recent_top_num=recent_top_num
        
        self.query_rank = query_rank
        self.query_recent_tokens=query_recent_tokens
        self.key_no_rope=key_no_rope
        # kv cache parameters
        self.kv_cache_dynamic = kv_cache_dynamic     
        self.kv_cache_eviction = kv_cache_eviction  
        self.stage_eviction=stage_eviction
        self.capacity=capacity
        self.raw_position_select = raw_position_select
        # position shift parameters
        self.position_shift = position_shift
        self.shift_factor=shift_factor
        self.interval_shift=interval_shift
        # positional_sorting
        self.positional_sorting=positional_sorting
        self.rank_ascend=rank_ascend
        # NTK
        self.NTK_aware = NTK_aware
        # base parameters
        self.delete_context_prompt=delete_context_prompt
        self.model_name = model_name
        self.data_name_full = data_name
        self.special_token=special_token
        self.attn_avg=attn_avg
        self.attn_implementation = attn_implementation
        self.model2prompt = model2prompt
        self.data_name=dataset
        self.tokenizer = model.tokenizer 
        self.parallel_pattern = parallel_pattern
        if data_name == "longbench":
            self.datas = self.load_datas(data_file)
        elif data_name == "infinitebench":
            self.datas = self.load_datas_infinitebench(data_file)
        self.templates = templates
        self.model = model
        self.base_random_seed = random_seed
        self.n_windows = n_windows
        self.context_max_len= context_max_len
        self.revise_single = False
        self.raw_model_max_len = raw_model_max_len
        self.prompt_method = prompt_method
        self.accelerator = accelerator
        self.model_class = model_class
        self.Truncation_Method = Truncation_Method
        logger.info(f"Truncation_Method is {Truncation_Method}")
        
    
    def build_chat(self,prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt
    
    def load_datas_infinitebench(self, data_file: str) -> None:
        logger.info("Loading data...")
        prompts_all = []
        inputs = []
        contexts = []
        answerss = []
        ids = []
        languages = []
        datasets = []
        optionss = []
        input_max_len = 0
        all_classess = []
        lengths = []
        
        def iter_jsonl(fname, cnt=None):
            i = 0
            with open(fname, "r") as fin:
                for line in fin:
                    if i == cnt:
                        break
                    yield json.loads(line)
                    i += 1
        
        def load_data(data_name: str, data_dir: str = "/home/avnet/xiongjing/sjh/parallel_window/"):
            fname = Path(data_name)
            return list(iter_jsonl(fname))
        
        examples = load_data(data_file)
        logger.info(f"type(examples): {type(examples)}")
        logger.info(f"len(examples): {len(examples)}")
        logger.info(f"answer: {examples[4]['answer']}")
        logger.info(f"examples[0].keys(): {examples[0].keys()}")
        logger.info("Finish loading InfiniteBench")
        
        def get_answer(eg: dict, data_name: str):
            if data_name in ["code_debug", "longbook_choice_eng"]:
                OPTIONS = "ABCD"
                if isinstance(eg["answer"], str):
                    ret = [eg["answer"], OPTIONS[eg['options'].index(eg["answer"])]]
                elif isinstance(eg["answer"], list):
                    if len(eg["answer"]) == 1:
                        ret = [eg["answer"][0], OPTIONS[eg['options'].index(eg["answer"][0])]]
                    elif len(eg["answer"]) == 2 and eg["answer"][1] in ['A', 'B', 'C', 'D']:
                        ret = eg['answer']
                    else:
                        raise ValueError
                else:
                    raise ValueError
                return ret

            return eg["answer"]
        
        def create_prompt(eg,data_name):
            template = self.model2prompt[data_name]
            if data_name == "code_run":
                find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
                func_call = find_result[0]
                func = func_call.split("(")[0]
                return template.format(
                    func=func,
                    func_call=func_call,
                    context=eg["context"],
                )
            elif data_name in ["code_debug", "code_debug_qa"]:
                code = eg["context"]
                if data_name == "code_debug":
                    return template.format(
                        context=code,
                        OPTION_A=eg["options"][0],
                        OPTION_B=eg["options"][1],
                        OPTION_C=eg["options"][2],
                        OPTION_D=eg["options"][3],
                    )
                return template.format(context=code)
            elif data_name == "longdialogue_qa_eng":
                script = eg["context"]
                prompt = template.format(context=script)
                return prompt
            elif data_name in [
                "longbook_choice_eng",
                "longbook_qa_eng",
                "longbook_sum_eng",
                "longbook_qa_chn",
            ]:
                book = eg["context"]
                if data_name == "longbook_choice_eng":
                    return template.format(
                        input=eg["input"],
                        context=book,
                        OPTION_A=eg["options"][0],
                        OPTION_B=eg["options"][1],
                        OPTION_C=eg["options"][2],
                        OPTION_D=eg["options"][3],
                    )
                elif data_name == "longbook_qa_eng":
                    return template.format(
                        input=eg["input"],
                        context=book,
                    )
                elif data_name == "longbook_sum_eng":
                    return template.format(context=book)
                elif data_name == "longbook_qa_chn":
                    return template.format(
                        input=eg["input"],
                        context=book,
                    )
                else:
                    raise ValueError
            elif data_name == "math_calc":
                return template.format(context=eg["context"])
            elif data_name == "math_find":
                prompt = eg['input']
                context = eg['context']
                find_result = re.findall(r"The .+ of", prompt)
                assert find_result, f"Cannot find the target number in {prompt}"
                target_number = find_result[0].lower()[:-3]
                prefix = f"What is {target_number} in the following list?"
                return template.format(
                    prefix=prefix,
                    context=context,
                    input=prompt,
                )

            # Default behavior if content key exists
            if "content" in eg:
                content = eg["content"]
                del eg["content"]
                eg["context"] = content

            format_dict = {
                "context": eg["context"],
                "input": eg["input"],
            }
            prompt = template.format(**format_dict)
        
        
            return prompt

        
        for example in examples:
            prompt = self.model2prompt[self.data_name]
            example["prompt"] = create_prompt(example, self.data_name)
            example["length"] = len(example["context"].split())
            example["all_classes"] = None
        
        input_max_len_string, query_max_len, context_max_len, maxid = 0, 0, 0, 0
        for i,example in enumerate(examples):
            # if i > 16:
            #     continue
            length_string = len(example["context"])
            length_query = self.tokenizer(example["input"], return_tensors="pt")["input_ids"].shape[1]
            length_context = self.tokenizer(example["context"], return_tensors="pt")["input_ids"].shape[1]
            if length_string > input_max_len_string: input_max_len_string = length_string
            if length_query > query_max_len: query_max_len = length_query
            if length_context > context_max_len : 
                context_max_len = length_context
                maxid = i
            
            prompts_all.append(example["prompt"])
            inputs.append(example["input"])
            languages.append("None")
            contexts.append(example["context"])
            answerss.append(get_answer(example, self.data_name))
            # print("get_answer(example, args.dataset)",get_answer(example, self.data_name))
            optionss.append(example["options"])
            datasets.append(self.data_name)
            all_classess.append(example["all_classes"])
            lengths.append(example["length"])
            ids.append(example["id"])
        logger.info(f"input_max_len_string: {input_max_len_string}")
        logger.info(f"query_max_len: {query_max_len}")
        logger.info(f"context_max_len: {context_max_len}")
        logger.info(f"id: {maxid}")
        # assert 1==0
        combined_data = [
            {
                "prompt": p, "input": i, "context": c, "answers": a, 
                "options": op, "_id": id, "dataset": dataset,
                "all_classes": ac, "length":l,  "language": lang,
            }
            for p, i, c, a, op, id, dataset, ac,l, lang in zip(
                prompts_all, inputs, contexts, answerss, optionss, ids,datasets,
                all_classess,lengths,languages,
            )
        ]
        # assert 1==0
        return combined_data
    
    def load_datas(self, data_file: str) -> None:
        test_data = []
        prompts_all = []
        inputs = []
        contexts = []
        answerss = []
        lengths = []
        datasets = []
        languages = []
        all_classess = []
        _ids = []
        input_max_len = 0
        input_max_len_string = 0
        query_max_len = 0
        context_max_len = 0
        with open(data_file) as fp:
            for line in fp:
                example = json.loads(line)
                length = example["length"]
                length_string = len(example["context"])
                length_query = self.tokenizer(example["input"], return_tensors="pt")["input_ids"].shape[1]
                length_context = self.tokenizer(example["context"], return_tensors="pt")["input_ids"].shape[1]
                if length > input_max_len: input_max_len = length
                if length_string > input_max_len_string: input_max_len_string = length_string
                if length_query > query_max_len: query_max_len = length_query
                if length_context > context_max_len : context_max_len = length_context
                # template = self.model2prompt[self.data_name]
                # prompt = template.format(**example)
                # if "llama2" in self.model_name or "llama-2" in self.model_name:
                #     prompt = self.build_chat(prompt)
                # example["prompt"] = prompt
                test_data.append(example)
        
        logger.info(f"Max Length is {input_max_len}")
        logger.info(f"Max Length string is {input_max_len_string}")
        logger.info(f"query_max_len tokens is {query_max_len}")
        logger.info(f"length_context_len tokens is {context_max_len}")
        if self.parallel_pattern == "test":
            assert 1==0
        for example in test_data:
            # prompts_all.append(example["prompt"])
            inputs.append(example["input"])
            contexts.append(example["context"])
            answerss.append(example["answers"])
            lengths.append(example["length"])
            datasets.append(example["dataset"])
            languages.append(example["language"])
            all_classess.append(example["all_classes"])
            _ids.append(example["_id"])
        logger.info("Finish loading dataset")
        
        combined_data = [
            {#"prompt": p,
                "input": i, "context": c, "answers": a, 
                "length": l, "dataset": d, "language": lang, 
                "all_classes": ac, "_id": id
            }#p,prompts_all
            for i, c, a, l, d, lang, ac, id in zip(
                inputs, contexts, answerss, lengths, 
                datasets, languages, all_classess, _ids
            )
        ]

        return combined_data
        
    def truncate_text(self,batch_size,batch_data) -> str:
        tokenizer = self.tokenizer
        
        
        batch_contexts = [item["context"] for item in batch_data]
        batch_inputs = [item["input"] for item in batch_data]
        new_prompt = batch_contexts
        
        # 确定是否启用并行窗口，采取不同策略
        if "default" in self.parallel_pattern:
            template = self.model2prompt[self.data_name]
            prompt = template.format(**batch_data[0])
            # prompt = batch_data[0]['prompt']
            # if "llama2" in self.model_name or "llama-2" in self.model_name:
            #     prompt = self.build_chat(prompt)
            new_prompt = [prompt]
        # logger.debug(f"new_prompt:{new_prompt}")
        logger.info("context length string:{}".format(len(new_prompt[0])))
        tokenized_prompts = self.tokenizer(new_prompt,
                                          padding="longest", 
                                          return_tensors="pt", 
                                          add_special_tokens=True,
                                          ).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        # torch.set_printoptions(edgeitems=100000, linewidth=200)
        # print(batch_input_ids)
        attention_mask = tokenized_prompts.attention_mask
        actual_lengths = attention_mask.sum(dim=1)
        max_len = actual_lengths.max().item()
        padding_len = max_len - actual_lengths
        if batch_size == 1:
            # print("self.Truncation_Method:{}".format(self.Truncation_Method))
            # assert 1==0
            if len(batch_input_ids[0]) > self.raw_model_max_len and self.Truncation_Method!="WO_Truncation"\
                and "default" in self.parallel_pattern:
                logger.info("fullkv truncation")
                print("raw context tokens length: {}".format(len(batch_input_ids[0])))
#                assert 1==0
                half = int(self.raw_model_max_len/2)
                truncation_prompts = [tokenizer.decode(batch_input_ids[i][padding_len[i]:padding_len[i]+half], skip_special_tokens=True)+
                          tokenizer.decode(batch_input_ids[i][-half:], skip_special_tokens=True) 
                          for i in range(len(batch_input_ids))]
#                assert 1==0
            else:
                if self.Truncation_Method=="WO_Truncation" :
                    # adaptive n windows
                    critical_length = 37000
                    
                    query_long_datasets = ["repobench-p","triviaqa"]
                    query_middle_datasets = ["samsum","passage_retrieval_en"]
                    if self.data_name in query_long_datasets:
                        self.context_max_len = 2500
                        logger.info(f"updata context_max_len: {self.context_max_len}")
                    elif self.data_name in query_middle_datasets:
                        self.context_max_len = 2800
                        logger.info(f"updata context_max_len: {self.context_max_len}")
                    
                    if self.kv_cache_eviction or self.stage_eviction or self.dynamic_window:
                        critical_length = torch.iinfo(torch.long).max
                    
                    logger.info(f"raw context tokens length: {len(batch_input_ids[0])}")
                    logger.info(f"critical_length: {critical_length}")
                    
                    if self.reduce_length and not self.revise_single: 
                        self.revise_single = True
                        if self.data_name not in query_long_datasets or self.data_name not in query_middle_datasets:
                            if self.reduce_factor == 0:
                                self.context_max_len = self.context_max_len // 2  # (1800)
                            else:
                                self.context_max_len = self.reduce_factor
                        else:
                            if self.context_max_len > self.reduce_factor:
                                self.context_max_len = self.reduce_factor    
                    logger.info(f"self.context_max_len: {self.context_max_len}")
                    # 划分窗口截断 最大context token数目限制在37k
                    if len(batch_input_ids[0]) <= critical_length:
                        total_len = batch_input_ids[0].shape[0]
                        # print("total_len:{}".format(total_len))
                        # assert 1==0
                        adaptive_n_windows = math.ceil(total_len / self.context_max_len)
                        #assert 1==0
                        logger.info(f"adaptive_n_windows in critical_length: {adaptive_n_windows}")
                        self.n_windows = adaptive_n_windows
                        truncation_prompts = [tokenizer.decode(batch_input_ids[i], skip_special_tokens=True) 
                              for i in range(len(batch_input_ids))]
                    else:
                        half = int(critical_length/2)
                        adaptive_n_windows = math.ceil(critical_length / self.context_max_len)
                        logger.info(f"adaptive_n_windows in full length: {adaptive_n_windows}")
                        self.n_windows = adaptive_n_windows
                        truncation_prompts = [tokenizer.decode(batch_input_ids[i][padding_len[i]:padding_len[i]+half], skip_special_tokens=True)+
                                  tokenizer.decode(batch_input_ids[i][-half:], skip_special_tokens=True) 
                                  for i in range(len(batch_input_ids))]
                    for layer in self.model.model.model.layers:
                        if self.rank_windows:
                            layer.self_attn.n_windows = min(self.n_windows,abs(self.topk_windows))
                            self.model.model.n_windows = min(self.n_windows,abs(self.topk_windows))
                        else:
                            layer.self_attn.n_windows = adaptive_n_windows
                            self.model.model.n_windows = adaptive_n_windows
                    self.model.n_windows = adaptive_n_windows
                else:
                    truncation_prompts = [tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)
                                  for i in range(len(batch_input_ids))]
                
        total_len = len(truncation_prompts[0])
        logger.info("after truncation context length string:{}".format(total_len))
        window_size = int(total_len/self.n_windows) # 平均每个窗口长度
        logger.info(f"window_size string: {window_size}")
#        assert 1==0
        per_windows_prompt = [truncation_prompts[0][i: i + window_size] for i in range(0, len(truncation_prompts[0]), window_size)]
        if len(truncation_prompts[0])-window_size*self.n_windows > 0:
            per_windows_prompt[-2]=per_windows_prompt[-2]+truncation_prompts[0][-(len(truncation_prompts[0])-window_size*self.n_windows):]
            per_windows_prompt = per_windows_prompt[:-1]
        
        # 对context加入prompt 根据pattern 确定添加哪些prompt
        per_windows_prompt = self.prompt_design(per_windows_prompt,batch_inputs*len(per_windows_prompt),batch_data[0])
        # 添加prompt后总token数目一定大于context_max_len
        
        # logger.info(f"per_windows_prompt:{per_windows_prompt}")
        # logger.debug(f"len(per_windows_prompt):{len(per_windows_prompt)}")
        # logger.debug(f"len(per_windows_prompt[0]):{len(per_windows_prompt[0])}")
        # logger.debug(f"per_windows_prompt[0]):{len(per_windows_prompt[0])}")
        # assert 1==0
        return per_windows_prompt

    def create_prompt_new(self,raw_sample,batch_data):
        if self.data_name == "math_find":
            prompt = raw_sample["input"]
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            prefix = f"What is {target_number} in the following list?"
            raw_sample['prefix'] = prefix
        elif self.data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", raw_sample['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            raw_sample["func"] = func
            raw_sample["func_call"] = func_call
        elif self.data_name == "longbook_choice_eng" or self.data_name =="code_debug":
            raw_sample["OPTION_A"]=batch_data["options"][0],
            raw_sample["OPTION_B"]=batch_data["options"][1],
            raw_sample["OPTION_C"]=batch_data["options"][2],
            raw_sample["OPTION_D"]=batch_data["options"][3],
        
            
    
    def prompt_design(self, raw_prompt:List[str], quesiton:List[str], batch_data):
        revise_prompts = []
        my_templates = self.templates
        
        for i in range(len(raw_prompt)):
            # 处理特殊数据集
            raw_sample = {}
            if "every_window_query_input_no_query" in self.parallel_pattern and "anchor" not in self.parallel_pattern:
                raw_sample["context"] = raw_prompt[i]
                raw_sample["input"] = quesiton[i]
                self.create_prompt_new(raw_sample,batch_data)
                revise_prompt = my_templates["all"].format(**raw_sample)
            elif "every_window_no_query_input_query" in self.parallel_pattern:
                raw_sample["context"] = raw_prompt[i]
                raw_sample["input"] = quesiton[i]
                self.create_prompt_new(raw_sample,batch_data)
                revise_prompt = my_templates["context"].format(**raw_sample)
            elif "every_window_query_input_query" in self.parallel_pattern and "anchor" not in self.parallel_pattern:
                
                raw_sample["context"] = raw_prompt[i]
                raw_sample["input"] = quesiton[i]
                self.create_prompt_new(raw_sample,batch_data)
                revise_prompt = my_templates["all"].format(**raw_sample)
            elif "every_window_query_input_query_anchor" in self.parallel_pattern:
                
                raw_sample["context"] = raw_prompt[i]
                raw_sample["input"] = quesiton[i]
                self.create_prompt_new(raw_sample,batch_data)
                if "new" in self.parallel_pattern: # 使用id
                    revise_prompt = {}
                    revise_prompt["context"] = my_templates["context"].format(**raw_sample)
                    revise_prompt["input"] = my_templates["question"].format(**raw_sample)
                else:
                    revise_prompt = my_templates["all_anchor"].format(**raw_sample)
            elif "default" in self.parallel_pattern:
                logger.info("len(raw_prompt):{}".format(len(raw_prompt)))
                assert len(raw_prompt) == 1
                revise_prompt = raw_prompt[i]
            else:
                revise_prompt = raw_prompt[i]
                
            # logger.info("len(revise_prompt:){}".format(len(revise_prompt)))
            revise_prompts.append(revise_prompt)
        
        
        return revise_prompts
    
    def get_predicted(self,eval_batch_size=1,output_max_len=32):
        accelerator = self.accelerator
        tokenizer = self.tokenizer
        raw_position_select = self.raw_position_select
        
        logger.info("get_predicted begin")
        with accelerator.split_between_processes(self.datas) as split_data:
            results=dict(outputs=[], num_tokens=0, first_token_time=0)
            split_data = list(split_data)
            for i in tqdm(range(0, len(split_data), eval_batch_size)):
                batch_data = split_data[i:i+eval_batch_size]
        
                # batch_prompts = [item["prompt"] for item in batch_data]
                batch_inputs = [item["input"] for item in batch_data]
                batch_contexts = [item["context"] for item in batch_data]
                batch_answerss = [item["answers"] for item in batch_data]
                batch_lengths = [item["length"] for item in batch_data]
                batch_datasets = [item["dataset"] for item in batch_data]
                batch_languages = [item["language"] for item in batch_data]
                batch_all_classess = [item["all_classes"] for item in batch_data]
                batch__ids = [item["_id"] for item in batch_data]
                
                # 在这里分配窗口
                self.create_prompt_new(batch_data[0],batch_data[0])
                per_windows_prompt = self.truncate_text(eval_batch_size,batch_data)
                logger.info(f"window_num: {len(per_windows_prompt)}")
                # assert 1==0
                
                question = self.templates["question"].format(**batch_data[0])
                if self.Truncation_Method=="WO_Truncation":
                   output = self.model.pcw_generate_longbench(per_windows_prompt,output_max_len,self.parallel_pattern,
                                                              question=question, context_max_len=self.context_max_len,
                                                              raw_model_max_len=self.raw_model_max_len,
                                                              adaptive_n_windows = self.n_windows,
                                                              raw_position_select=raw_position_select,
                                                              recent_token=self.recent_token,
                                                              )
                else:
                   output = self.model.pcw_generate_longbench(per_windows_prompt,output_max_len,self.parallel_pattern,
                                                              question=question,  
                                                              context_max_len=self.context_max_len,
                                                              raw_model_max_len=self.raw_model_max_len,
                                                              raw_position_select=raw_position_select,
                                                              recent_token = self.recent_token,
                                                             )
                # print("results:{}".format(output))
                batch_generations = [output]
                for j in range(len(batch_contexts)):
                    example = {}
                    # example["prompt"] = batch_prompts[j]
                    example["input"] = batch_inputs[j]
                    example["context"] = batch_contexts[j]
                    example["answers"] = batch_answerss[j]
                    example["pred"] = batch_generations[j]
                    example["length"] = batch_lengths[j]
                    
                    example["dataset"] = batch_datasets[j]
                    example["language"] = batch_languages[j]
                    example["all_classes"] = batch_all_classess[j]
                    example["_id"] = batch__ids[j]
                    results["outputs"].append(example)
                    results["num_tokens"] += len(batch_generations[j])
                    
            results = [results]
        self.accelerator.wait_for_everyone()
        results_gathered = gather_object(results)
        
        # self.accelerator.wait_for_everyone()
        return results_gathered

    # run_experiment
    def run_experiment(self, batch_size: int = 1,output_max_len:int = 32):
        accelerator = self.accelerator
        model_name = self.model_name
        model_class=self.model_class
        
        if self.prompt_method == "complex_cot_pcw_multi_windows" or \
            self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
            results_gathered=self.get_predicted(eval_batch_size=batch_size,output_max_len=output_max_len)
        
        
        import os
        datasets_name = self.data_name
        model_name = model_name.split('/')[-1]
        logger.debug(f"datasets_name:{datasets_name}")
        
        if self.Truncation_Method=="WO_Truncation":
            self.n_windows = "adaptive"
        if not self.kv_cache_eviction and not self.stage_eviction:
            self.capacity = "full"
        if accelerator.is_main_process:
            if self.data_name_full=="longbench":
                output_path = os.path.join("results/longbench_0118", f"{model_name}", datasets_name)
            elif self.data_name_full=="infinitebench":
                output_path = os.path.join("results/infinitebench_0117", f"{model_name}", datasets_name)
            else:
                output_path = os.path.join("results/others", f"{model_name}", datasets_name)
            os.makedirs(output_path, exist_ok=True)
            if self.parallel_pattern == "every_window_query_input_query_anchor_new":
                parallel_pattern = "anchorNew"
            else:
                parallel_pattern = self.parallel_pattern
            output_datapath = os.path.join(output_path,
                                    # f"{self.parallel_pattern}_windows_{self.n_windows}_{model_class}"+\
                                    f"{parallel_pattern}_"+\
                                    # f"{kv_cache}_raw_position_{int(self.raw_position_select)}_"+\
                                    # f"{kv_cache}_"+ \
                                    # f"winDynamic_{int(self.dynamic_window)}_"+\
                                    f"queryRank_{int(self.query_rank)}_" +\
                                    # f"rectok_{int(self.query_recent_tokens)}_" +\
                                    # f"rank_{int(self.rank_windows)}_" +\
                                    # f"cache_{int(self.rank_cache)}_" +\
                                    f"topk_{self.topk_windows}_" +\
                                    # f"ascend_{int(self.window_ascend)}_" +\
                                    # f"decom_{self.decomposition_factor}_" +\
                                    # f"recem_{int(self.get_recent_attn)}_" +\
                                    # f"num_{self.recent_top_num}_" +\
                                    # f"reduce_{int(self.reduce_length)}_" +\
                                    # f"tokenNum_{self.reduce_factor}_" +\
                                    f"calibStage_{self.calibration_stage}_" +\
                                    f"calibMode_{self.calibration_mode}_" +\
                                    f"delPro_{int(self.delete_context_prompt)}_" +\
                                    f"kv_pre_{int(self.kv_cache_eviction)}_"+\
                                    f"stage_{int(self.stage_eviction)}_"+\
                                    f"generate_{int(self.kv_cache_dynamic)}_" +\
                                    f"winsize{self.recent_token}_" +\
                                    f"cap_{self.capacity}_" +\
                                    f"keyNR_{int(self.key_no_rope)}_" +\
                                    f"poShift{int(self.position_shift)}_" +\
                                    f"factor_{self.shift_factor}_" +\
                                    f"interval_{self.interval_shift}_" +\
                                    f"poSort_{self.positional_sorting}_" +\
                                    f"ascend_{int(self.rank_ascend)}_" +\
                                    # f"NTK_{int(self.NTK_aware)}" +\
                                    f"spe_{int(self.special_token)}_" +\
                                    f"atAvg_{int(self.attn_avg)}" +\
                                    # f"_{self.attn_implementation}" +\
                                    f".json"
                                    )
            logger.info(f"output_datapath: {output_datapath}")
            fout = open(output_datapath,"w")
            for result_list in results_gathered:
                for example in result_list["outputs"]:
                    fout.write(json.dumps(example) + "\n")
        
