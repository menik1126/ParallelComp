import logging
import random
from typing import List, Dict

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

STOP_SEQUENCE = '\n'

class ExperimentManager_longbench:
    def __init__(self, data_file: str, model: PCWModelWrapper, random_seed: int = 42, n_windows: int = 1, 
                 max_token_length: int = 1024,
                 prompt_method:str=None,model_class:str=None,parallel_pattern:str=None,
                 model_name:str=None,dataset:str=None,model2prompt=None,templates:Dict=None,
                 accelerator: Accelerator=None,
                 ):   
        self.model_name = model_name
        self.model2prompt = model2prompt
        self.data_name=dataset
        self.datas = self.load_datas(data_file)
        self.templates = templates
        
        self.model = model
        self.base_random_seed = random_seed
        self.n_windows = n_windows
        self.model_max_len= max_token_length
        self.tokenizer = model.tokenizer
        self.prompt_method = prompt_method
        self.accelerator = accelerator
        self.model_class = model_class
        self.parallel_pattern = parallel_pattern
        
    
    def build_chat(self,prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt
    
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
        with open(data_file) as fp:
            for line in fp:
                example = json.loads(line)
                length = example["length"]
                if length > input_max_len: input_max_len = length
                # template = self.model2prompt[self.data_name]
                # prompt = template.format(**example)
                # if "llama2" in self.model_name or "llama-2" in self.model_name:
                #     prompt = self.build_chat(prompt)
                # example["prompt"] = prompt
                test_data.append(example)
        logger.info(f"Max Length is {input_max_len}")
        
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
            for  i, c, a, l, d, lang, ac, id in zip(
                inputs, contexts, answerss, lengths, 
                datasets, languages, all_classess, _ids
            )
        ]

        return combined_data
        
    def truncate_text(self,batch_size,batch_data) -> str:
        model_max_len = self.model_max_len
        tokenizer = self.tokenizer
        n_windows=self.n_windows
        
        batch_contexts = [item["context"] for item in batch_data]
        batch_inputs = [item["input"] for item in batch_data]
        
        tokenized_prompts = self.tokenizer(batch_contexts,
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
            if len(batch_input_ids[0]) > model_max_len:
                half = int(model_max_len/2)
                truncation_prompts = [tokenizer.decode(batch_input_ids[i][padding_len[i]:padding_len[i]+half], skip_special_tokens=True)+
                          tokenizer.decode(batch_input_ids[i][-half:], skip_special_tokens=True) 
                          for i in range(len(batch_input_ids))]
                
            else:
                truncation_prompts = [tokenizer.decode(batch_input_ids[i], skip_special_tokens=True) 
                          for i in range(len(batch_input_ids))]
            
            window_size = int(len(truncation_prompts[0])/n_windows)
            per_windows_prompt = [truncation_prompts[0][i: i + window_size] for i in range(0, len(truncation_prompts[0]), window_size)]
            if len(truncation_prompts[0])-window_size*n_windows > 0:
                per_windows_prompt[-2]=per_windows_prompt[-2]+truncation_prompts[0][-(len(truncation_prompts[0])-window_size*n_windows):]
                per_windows_prompt = per_windows_prompt[:-1]
        

        per_windows_prompt = self.prompt_design(per_windows_prompt,batch_inputs*len(per_windows_prompt))
        # logger.info(f"per_windows_prompt:{per_windows_prompt}")
        # logger.debug(f"len(per_windows_prompt):{len(per_windows_prompt)}")
        # logger.debug(f"len(per_windows_prompt[0]):{len(per_windows_prompt[0])}")
        # logger.debug(f"per_windows_prompt[0]):{len(per_windows_prompt[0])}")
        # assert 1==0
        return per_windows_prompt

    def prompt_design(self, raw_prompt:List[str], quesiton:List[str]):
        revise_prompts = []
        my_templates = self.templates
        
        for i in range(len(raw_prompt)):
            if self.parallel_pattern == "every_window_query_input_no_query" :
                raw_sample = {}
                raw_sample["context"] = raw_prompt[i]
                raw_sample["input"] = quesiton[i]
                revise_prompt = my_templates["all"].format(**raw_sample)
            elif self.parallel_pattern == "every_window_no_query_input_query":
                raw_sample = {}
                raw_sample["context"] = raw_prompt[i]
                raw_sample["input"] = quesiton[i]
                # logger.debug(f"{my_templates['context']}")
                revise_prompt = my_templates["context"].format(**raw_sample)
            elif self.parallel_pattern == "every_window_query_input_query":
                raw_sample = {}
                raw_sample["context"] = raw_prompt[i]
                raw_sample["input"] = quesiton[i]
                revise_prompt = my_templates["all"].format(**raw_sample)
            else:
                revise_prompt = raw_prompt[i]
            
            revise_prompts.append(revise_prompt)
        
        
        return revise_prompts
    
    def get_predicted(self,eval_batch_size=1,output_max_len=32):
        accelerator = self.accelerator
        tokenizer = self.tokenizer
        
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

                per_windows_prompt = self.truncate_text(eval_batch_size,batch_data)
                logger.info(f"len(per_windows_prompt):{len(per_windows_prompt)}")
                # assert 1==0
                question = self.templates["question"].format(**batch_data[0])
                
                output = self.model.pcw_generate_longbench(per_windows_prompt,output_max_len,self.parallel_pattern,question=question)
                print("results:{}".format(output))
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
        results_gathered = gather_object(results)
        self.accelerator.wait_for_everyone()
        return results_gathered

    # run_experiment
    def run_experiment(self, batch_size: int = 1,output_max_len:int = 32):
        accelerator = self.accelerator
        model_name = self.model_name
        model_class=self.model_class
        
        if self.prompt_method == "complex_cot_pcw_multi_windows":
            results_gathered=self.get_predicted(eval_batch_size=batch_size,output_max_len=output_max_len)
        
        datasets_name = self.data_name
        if accelerator.is_main_process:
            import os
            os.makedirs(os.path.join("results/longbench", f"{model_name}", datasets_name), exist_ok=True)
            fout = open(os.path.join("results/longbench", f"{model_name}", datasets_name,
                                     f"{self.parallel_pattern}_windows_{self.n_windows}_{model_class}.json"), "w")
            for result_list in results_gathered:
                for example in result_list["outputs"]:
                    fout.write(json.dumps(example) + "\n")
        

class ExperimentManager:
    def __init__(self, test_df: pd.DataFrame, train_df: pd.DataFrame, model: PCWModelWrapper,
                 labels: List[str] = None, random_seed: int = 42, subsample_test_set: int = 2000,
                 n_shots_per_window: int = None, prompt_method: str = None, output_json:str = None, n_windows: int = 1, accelerator=None, sample_method=None, sample_number=None, extra_sample_number=None):
        if subsample_test_set < len(test_df):
            np.random.seed(random_seed)
            test_df = test_df.sample(subsample_test_set)
        self.prompt_method = prompt_method
        print("self.prompt_method:{}".format(self.prompt_method))
        self.test_df = test_df
        self.n_windows = n_windows  # 窗口数量

        self.train_df = train_df
        self.model = model
        self.base_random_seed = random_seed
        self.n_shots_per_window = n_shots_per_window
        self.tokenizer = model.tokenizer
        self.output_json = output_json
        self.multi_gpus = torch.cuda.device_count() > 1
        self.accelerator = accelerator
        
        self.sample_method = sample_method
        self.sample_number = sample_number
        self.extra_sample_number = extra_sample_number
        

        with open("datasets/gsm8k/complex_cot.txt", 'r', encoding='utf-8') as file:
            prompt = file.read()
            # prompts =  prompt.split("\n\n")
            # print("prompts[0]:{}".format(prompts[0]))
            # print("len(prompts):{}".format(len(prompts)))
        self.complex_cot = prompt
        #print("self.complex_cot:{}".format(self.complex_cot))
        if self.prompt_method == "other" :
            # self.max_n_tokens = self.tokenizer.model_max_length
            self._initialize_labels_and_logit_processor(labels, self.prompt_method)


    def _initialize_labels_and_logit_processor(self, labels: List[str], prompt_method: str) -> None:
        """
            函数的输入和用途
                输入: labels (一个字符串列表)
                表示分类任务的所有可能标签。
                
                用途:
                编码标签为模型可以处理的格式。
                优化标签的长度以减少生成复杂度。
                为生成任务设置逻辑约束，限制模型输出为预定义的标签。
        """
        _logger.info(f"Provided labels: {labels}")
        labels_tokens = encode_labels(self.tokenizer, labels)
        labels_tokens_array = self.minimize_labels_tokens(labels_tokens)
        _logger.info(f"Provided labels average n_tokens: {np.round(np.mean([len(lt) for lt in labels_tokens]), 3)}")
        # we fix the labels accordingly in the test set:
        shorten_label_tokens = [t[t != self.tokenizer.eos_token_id].tolist() for t in labels_tokens_array]
        _logger.info(
            f"shortened labels average n_tokens: {np.round(np.mean([len(lt) for lt in shorten_label_tokens]), 3)}")
        # Moving the test set label tokens to their shorter version:
        map_labels = {old_label: self.tokenizer.decode(t).lstrip() for old_label, t in
                      zip(labels, shorten_label_tokens)}
        self.test_df[LABEL_TOKENS] = self.test_df[LABEL_TOKENS].map(map_labels)
        pad = len(max(shorten_label_tokens, key=len))
        labels_tokens_array = np.array(
            [i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in shorten_label_tokens])
        self.max_n_tokens = pad
        labels_tokens_array = self.pad_contained_labels_with_stop_seq(shorten_label_tokens, labels_tokens_array)
        self.logit_processor = RestrictiveTokensLogitsProcessor(restrictive_token_ids=labels_tokens_array,
                                                                eos_token_id=self.tokenizer.eos_token_id)
        self.possible_labels = set(map_labels.values())

    def minimize_labels_tokens(self, labels_tokens: List[List[int]]) -> npt.NDArray[int]:
        """
         Minimize the number of tokens per label to be the shortest possible unique one.
        """
        pad = len(max(labels_tokens, key=len))
        labels_tokens_array = np.array([i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in labels_tokens])
        for i, tokens in enumerate(labels_tokens):
            for j in range(len(tokens)):
                labels_with_shared_beginnings = np.sum(
                    np.all(labels_tokens_array[:, :j] == np.array(tokens[:j]), axis=1))
                if labels_with_shared_beginnings == 1:
                    labels_tokens_array[i, j:] = self.tokenizer.eos_token_id
                    break
        return labels_tokens_array

    def pad_contained_labels_with_stop_seq(self, labels_tokens: List, labels_tokens_array: npt.NDArray[int]) \
            -> npt.NDArray[int]:
        """
        In case we have two labels, where one label contains the other label (for example: "A" and "A B") we need
        to allow the restrictive decoding to produce the output "A". We support it by adding "\n" to the shorter label.
        """
        stop_seq_token_id = encode_stop_seq(self.tokenizer, STOP_SEQUENCE)
        for i, tokens in enumerate(labels_tokens):
            labels_with_shared_beginnings = np.sum(
                np.all(labels_tokens_array[:, :len(tokens)] == np.array(tokens), axis=1))
            if labels_with_shared_beginnings > 1:
                _logger.info(f"label{self.tokenizer.decode(tokens)} is the beginning of one of the other labels,"
                             f"adding stop sequence to its end")
                labels_tokens_array[i, len(tokens)] = stop_seq_token_id
        return labels_tokens_array

    def _set_random_seed(self, random_seed: int) -> None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    def get_few_shots_acc(self, windows_few_shot: List[str], few_shots_prompts: str=None) -> float:
        returns = self.get_predicted_labels(windows_few_shot, few_shots_prompts=few_shots_prompts)
        if self.prompt_method == "complex_cot" or self.prompt_method == "complex_cot_pcw" or self.prompt_method == "complex_cot_pcw_pre_process_window_cache" or self.prompt_method == "complex_cot_pcw_multi_windows" or self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache"\
            :
            if self.multi_gpus:
                if self.accelerator.is_main_process:
                    with open(self.output_json, "w") as f:
                        json.dump(returns, f)
            else:
                with open(self.output_json, "w") as f:
                    json.dump(returns, f)
        else:
            return self.calc_acc(returns)

    def get_predicted_labels(self, windows_few_shots: List[str], few_shots_prompts: str=None) -> List[str]:
                    
        predicted_labels = []
        all_preds = []
        i = 0 
        
        
        if self.multi_gpus:
            self.accelerator.wait_for_everyone()
            all_data = (self.test_df[[PROMPTS, "question", "gold_reasoning"]]).values.tolist() #question, gold_reasoning
            #print("len of all_data:{}".format(len(all_data)))
            with self.accelerator.split_between_processes(all_data) as prompts:
                # print("len of test data:{}".format(len(prompts)))
                # print(f"Hello this is GPU {self.accelerator.process_index}")

                for q in tqdm(prompts):            
                    if self.prompt_method == "complex_cot" or self.prompt_method == "complex_cot_pcw" or self.prompt_method == "complex_cot_pcw_multi_windows" or self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
                        cot = self.predict_label(q[0], few_shots_prompts=few_shots_prompts)
                        all_preds.append({'question':q[1],'answer': q[2],'pred': cot})
                    elif  self.prompt_method == "complex_cot_pcw_pre_process_window_cache":
                        cot = self.predict_label(q[0], windows_cache=windows_cache)
                        all_preds.append({'question':q[1],'answer': q[2],'pred': cot})
                    else:
                        predicted_label = self.predict_label(TEXT_BETWEEN_SHOTS + q[1], windows_cache=windows_cache)
                        all_preds.append(predicted_label)
                    i = i + 1
                    # if i==5:
                    #     break 
            
                all_preds = gather_object(all_preds)
                return all_preds
        else:
            for q in tqdm(self.test_df[PROMPTS]):            
                    if self.prompt_method == "complex_cot" or self.prompt_method == "complex_cot_pcw" or self.prompt_method == "complex_cot_pcw_multi_windows" or self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
                        # {'question':question,'answer': backup[i]['answer'],'pred': pred}
                        cot = self.predict_label(q, few_shots_prompts=few_shots_prompts)
                        all_preds.append({'question':self.test_df.loc[i, "question"],'answer': self.test_df.loc[i, "gold_reasoning"],'pred': cot})
                    elif self.prompt_method == "complex_cot_pcw_pre_process_window_cache":
                        cot = self.predict_label(q, few_shots_prompts=few_shots_prompts, windows_cache=windows_cache)
                        all_preds.append({'question':self.test_df.loc[i, "question"],'answer': self.test_df.loc[i, "gold_reasoning"],'pred': cot})
                    else:
                        predicted_label = self.predict_label(TEXT_BETWEEN_SHOTS + q, windows_cache=windows_cache)
                        all_preds.append(predicted_label)
                    i = i + 1
            if self.prompt_method == "other":        
               assert set(predicted_labels).issubset(self.possible_labels)                
            return all_preds
            
    def predict_label(self, task_text: str, windows_cache: str=None, few_shots_prompts: str=None) -> str:
        """
           这行代码的作用是检查 task_text 字符串的末尾是否有空格。如果末尾有空格，代码会抛出一个 AssertionError，并显示提示信息 "prompt ends with a space!"。
           rstrip() 返回一个新的字符串，去掉了末尾的空格和换行符，结果是 'Hello, World!'。
           
           cache: 是返回的长下文编码
        """

        if self.prompt_method == "complex_cot" or self.prompt_method == "complex_cot_pcw" or self.prompt_method == "complex_cot_pcw_pre_process_window_cache" or self.prompt_method == "complex_cot_pcw_multi_windows" or self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
            
            res = self.model.pcw_generate(task_text=task_text,
                                          contexts_cache=windows_cache,
                                          temperature=0.0,
                                          max_new_tokens=300, 
                                          few_shots_prompts=few_shots_prompts,
                                          do_sample=False,
                                          num_return_sequences=1
                                         )
            return res
        else:
            res = self.model.pcw_generate(task_text=task_text,
                                        contexts_cache=windows_cache,
                                        restrictive_logit_preprocessor=self.logit_processor,
                                        temperature=0,
                                        max_new_tokens=self.max_n_tokens)

            return res.lstrip().strip(STOP_SEQUENCE)

    def calc_acc(self, predicted_labels: List) -> float:
        predicted_labels = pd.Series(predicted_labels, index=self.test_df.index)
        acc = np.mean(predicted_labels == self.test_df[LABEL_TOKENS])
        _logger.info(f"accuracy = {np.round(acc, 3)}")
#        assert 1==0
        return acc

    def run_experiment_across_shots(self, n_shots_to_test: List[int], n_runs: int,
                                    too_long_patience: float = 0.2):
        """
            n_shots_to_test：  n_shots = [i * n_shots_per_window for i in n_windows]
            n_runs: Number of times experiments are repeated for every number of windows  重复实验次数
        """
        accuracies = np.zeros((len(n_shots_to_test), n_runs))
        for i, n_shots in enumerate(tqdm(n_shots_to_test)):    # 分别在两种setting(两种不同上下文窗口长度)下做
            """
                n_shots： 代表所有数据的数量
            """
            _logger.info(f"starting with n = {n_shots}")
            self._set_random_seed(self.base_random_seed + n_shots)

            n_errors = 0
            for j in tqdm(range(n_runs), desc="Progress"):  # 重复n_run次实验
                if self.prompt_method == "complex_cot" or self.prompt_method == "complex_cot_pcw" or \
                    self.prompt_method == "complex_cot_pcw_multi_windows" or \
                        self.prompt_method =="complex_cot_pcw_pre_process_window_cache"\
                            or self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
                    if self.sample_method == "sample":
                       few_shots_prompts = "\n\n".join(list(self.train_df[PROMPTS]))
                       #print("few_shots_prompts:{}".format(few_shots_prompts))
                       #assert 1==0
                    else: 
                       
                       few_shots_prompts = self.complex_cot #list(self.train_df.loc[few_shots_idx, PROMPTS])
                       all_prompts = list(self.train_df[PROMPTS])
                       if self.extra_sample_number>0:
                          all_prompts = all_prompts[:self.extra_sample_number]
                          few_shots_prompts = few_shots_prompts + "\n\n" + "\n\n".join(all_prompts)
                          print("few_shots_prompts:{}".format(few_shots_prompts))
                          #assert 1==0
                    

                else:
                    few_shots_idx = self.sample_n_shots(n_shots)   # 采集上下文示例
                    few_shots_prompts = list(self.train_df.loc[few_shots_idx, PROMPTS])
       
                # 上面都是在采集上下文示例, 包括根据token数量平衡窗口个数
                # print("self.n_shots_per_window:{}".format(self.n_shots_per_window))
                #print("len(few_shots_prompts) -2:{}".format(len(few_shots_prompts)))
                windows_few_shots = self.build_windows_few_shots_text(few_shots_prompts, self.n_shots_per_window)
                longest_window_n_tokens = max(n_tokens_in_prompt(self.tokenizer, window)
                                              for window in windows_few_shots)

                n_tokens_between_shots = n_tokens_in_prompt(self.tokenizer, TEXT_BETWEEN_SHOTS)
                # 上面都是在组织窗口长度
                

                # windows_few_shots
                if self.prompt_method == "complex_cot" or self.prompt_method == "complex_cot_pcw" or self.prompt_method == "complex_cot_pcw_multi_windows" or self.prompt_method == "complex_cot_pcw_pre_process_window_cache"\
                    or self.prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
                    self.get_few_shots_acc(windows_few_shots, few_shots_prompts=few_shots_prompts)
                else:
                    if ((longest_window_n_tokens + n_tokens_between_shots + self.test_df[N_TOKENS].max() + self.max_n_tokens) > self.model.context_window_size) :
                        """
                            这段代码的目的是检查当前生成的训练窗口（包含训练样本、分隔符、测试样本等内容）的总 token 数量是否超过了模型的上下文窗口大小（context_window_size）。
                            如果超过，记录警告信息，并重新尝试生成一个符合要求的训练窗口。以下是逐步解析：
                        """
                            
                        _logger.warning("Drawn training shots were too long, trying again")
                        n_errors += 1
                        print("n_errors:{}".format(n_errors))
                        assert n_errors <= too_long_patience * n_runs, "too many long inputs were drawn!"
                        continue
                    accuracies[i, j] = self.get_few_shots_acc(windows_few_shots, few_shots_prompts=few_shots_prompts)

        return accuracies

    def sample_n_shots(self, n_shots: int) -> npt.NDArray[int]:
        """
           在这里晚上上下文示例采样
           n_shots_per_window: 每个窗口最多可以容纳的样本数量
           n_shots:  i * n_shots_per_window 即总的窗口数量*每个窗口的上下文示例数量   #[i * n_shots_per_window for i in n_windows]
        """
        print("n_shots:{}".format(n_shots))
        few_shots_df = self.train_df.sample(n_shots)  # 从训练集里面采样示例, 而且是idx
        # print("few_shots_df:{}".format(few_shots_df))
        # print("len(few_shots_df):{}".format(len(few_shots_df)))
        assert few_shots_df.index.is_unique, "few shots samples were not unique!"
        window_size = self.n_shots_per_window or n_shots
        print("window_size:{}".format(window_size))
        #assert 1==0
        n_windows = int(len(few_shots_df) / window_size)   # 总的窗口数量
        print("n_windows:{}".format(n_windows))
        
        if not self.n_shots_per_window or n_windows == 1:
            return few_shots_df.index
        # balance_windows_sizes  这个不知道干啥的
        return self.balance_windows_sizes(n_windows, few_shots_df)
    
    def divide_n_windows(self, n_shots: int, n_windows: int) :
        """
            把多个上下文示例分到两个窗口里
        """
        print("n_shots:{}".format(n_shots))
        few_shots_df = self.train_df.sample(n_shots)  # 从训练集里面采样示例, 而且是idx
        # print("few_shots_df:{}".format(few_shots_df))
        # print("len(few_shots_df):{}".format(len(few_shots_df)))
        assert few_shots_df.index.is_unique, "few shots samples were not unique!"
        window_size = self.n_shots_per_window or n_shots
        print("window_size:{}".format(window_size))
        n_windows = int(len(few_shots_df) / window_size)   # 总的窗口数量
        print("n_windows:{}".format(n_windows))
        
        if not self.n_shots_per_window or n_windows == 1:
            return few_shots_df.index
        # balance_windows_sizes  这个不知道干啥的
        return [2,3]

    def balance_windows_sizes(self, n_windows: int, few_shots_df: pd.DataFrame) -> npt.NDArray[int]:
        """
           这是一个函数，用于平衡窗口大小和分布，从一个包含 token 数量（n_tokens）信息的 DataFrame 中选择样本，以优化分配到多个窗口的 token 总量。以下是逐步解释：
        """
        few_shots_df.sort_values(by=N_TOKENS, inplace=True, ascending=False)
        shape = (self.n_shots_per_window, n_windows)
        indexes = np.array(few_shots_df.index).reshape(shape)
        sizes = few_shots_df.loc[indexes.flatten()].n_tokens.values.reshape(indexes.shape)
        for i in range(1, self.n_shots_per_window):
            order = np.argsort((np.sum(sizes[:i, :], axis=0)))
            sizes[i, :] = sizes[i, order]
            indexes[i, :] = indexes[i, order]
        # shuffle the order in each window:
        for i in range(n_windows):
            np.random.shuffle(indexes[:, i])
        indexes = indexes.T.flatten()
        return indexes

    def build_windows_few_shots_text(self, few_shots_prompts: List, window_size: int) -> List[str]:
        """
            self.n_shots_per_window: 每个窗口有多少shot数据
        """
        print("self.n_shots_per_window:{}".format(window_size))
        if window_size is None:
            window_size = len(few_shots_prompts)
        if self.prompt_method == "other":
           return [TEXT_BETWEEN_SHOTS.join(few_shots_prompts[i: i + window_size]) for i in
                range(0, len(few_shots_prompts), window_size)]     # range(0, len(few_shots_prompts), window_size)：起点从 0 开始，每次步进 window_size，确保不遗漏元素。
           
        else: 
           few_shots_prompts = few_shots_prompts.split("\n\n")
           #print("few_shots_prompts list:{}".format(few_shots_prompts))
           #assert 1==0
           return [TEXT_BETWEEN_SHOTS_CoT.join(few_shots_prompts[i: i + window_size]) for i in
                range(0, len(few_shots_prompts), window_size)]
           
          

