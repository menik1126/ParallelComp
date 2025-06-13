import torch
from transformers import AutoConfig, LlamaTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM

from accelerate import Accelerator
from pcw_wrapper import PCWModelWrapper
from pcw_wrapper_batches import PCWModelWrapperBatches
from typing import Optional
from transformers.models.llama.modeling_llama import LlamaForCausalLM,LlamaRotaryEmbedding,LlamaConfig
import torch
# accelerator = Accelerator()
GPT2_WINDOW_SIZE = 1024
LLAMA_WINDOW_SIZE = 2048
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
from my_utils.logger import Logger
logger = Logger()
logger.set_console_level(logging.DEBUG)


def validate_model_name(model_name: str) -> None:
    assert 'llama' in model_name.lower() or 'gpt2' in model_name \
        or "qwen" in model_name.lower() or 'gemma' in model_name.lower(),\
        f"Unknown model: {model_name}"


def load_tokenizer(model_name: str,) -> PreTrainedTokenizerBase:
    if 'llama' in model_name.lower() or 'gemma' in model_name.lower() \
        or 'qwen' in model_name.lower():
        if model_name == 'seanmor5/tiny-llama-test' or 'decapoda-research' in model_name:  # debug mode:
            tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
            # In case you load those models, we must override an incorrect config:
            # see: https://huggingface.co/decapoda-research/llama-7b-hf/discussions/12
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("success load tokenizer")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # In our experiments we have added bos token to gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token=True)
    return tokenizer

from accelerate import dispatch_model

def load_pcw_wrapper(model_name: str, cache_dir: str = None, n_windows: int = 1, 
                     # base parameters
                     model_class: str = None, accelerator=None, 
                     parallel_pattern=None,
                     raw_model_max_len=None,special_token=None,
                     context_prompt=None,
                     # attention calibration
                     calibration_mode=0,calibration_stage=None,
                     # kv cache parameters
                     kv_cache_eviction=False,kv_cache_dynamic=False,stage_eviction=False,capacity=None, 
                     # windows parameters
                     topk_windows=None,
                     query_rank=False,
                     query_recent_tokens=0,
                     # other parameters
                     head_datas=None,
                     ) -> PCWModelWrapper:
    print("model_name:{}".format(model_name))
#    assert 1==0
    validate_model_name(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args = {
        "cache_dir": cache_dir
    }
    if  model_class == "modeling_llama_with_pcw_kv_cache_FlashAttention_longbench":
        from modeling_llama_with_pcw_kv_cache_FlashAttention_longbench import LlamaForCausalLMPCW
        model_obj = LlamaForCausalLMPCW
    elif  model_class == "modeling_gemma_with_pcw_kv_cache_FlashAttention_longbench":
        from modeling_gemma_with_pcw_kv_cache_FlashAttention_longbench import Gemma2ForCausalLMPCW
        model_obj = Gemma2ForCausalLMPCW
    elif model_class == "modeling_qwen2_with_pcw_kv_cache_FlashAttention_longbench":
        from modeling_qwen2_with_pcw_kv_cache_FlashAttention_longbench import Qwen2ForCausalLMPCW
        model_obj = Qwen2ForCausalLMPCW
    elif model_class == "modeling_llama":
       from transformers.models.llama.modeling_llama import LlamaForCausalLM
       model_obj = LlamaForCausalLM
    
    if "kv_cache" in model_class:
        model = model_obj.from_pretrained(model_name, capacity = capacity, 
                                         n_windows=n_windows, 
                                         kv_cache_eviction=kv_cache_eviction,
                                         kv_cache_dynamic=kv_cache_dynamic,
                                         stage_eviction=stage_eviction,
                                         calibration_mode=calibration_mode,
                                         calibration_stage=calibration_stage,
                                         parallel_pattern=parallel_pattern,
                                         head_datas=head_datas,
                                         attn_implementation="flash_attention_2",
                                         torch_dtype=torch.float16,
                                         device_map={"": accelerator.process_index},
                                        #  device_map = "auto",
                                         **model_args).eval()
        
        model.half()
    else:
        model = model_obj.from_pretrained(model_name,**model_args).eval()
        model.half()
    
    if hasattr(model, "module"):
       model = model.module
    context_window_size = LLAMA_WINDOW_SIZE

    tokenizer = load_tokenizer(model_name)
    # 1226
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 给model添加属性
    # if not multi_gpus:
    #     model = model.to(device)
    
    if "batches" in parallel_pattern:
        PCW = PCWModelWrapperBatches
    else:
        PCW = PCWModelWrapper
    return PCW(model, tokenizer, device, context_window_size,
                           # base parameters
                           n_windows=n_windows, 
                           model_name=model_name,
                           parallel_pattern=parallel_pattern,
                           raw_model_max_len=raw_model_max_len,special_token=special_token,
                           context_prompt=context_prompt,
                           # kv_cache parameters
                           capacity=capacity, kv_cache_eviction=kv_cache_eviction, stage_eviction=stage_eviction,
                           # windows parameters
                           topk_windows=topk_windows, 
                           query_rank=query_rank,
                           query_recent_tokens=query_recent_tokens,
                           # attention calibration
                           calibration_stage=calibration_stage,calibration_mode=calibration_mode,
                           )
