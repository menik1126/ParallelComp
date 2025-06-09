import torch
from transformers import AutoConfig, LlamaTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM

from accelerate import Accelerator
from modeling_gpt2_with_pcw import GPT2LMHeadPCW
from pcw_wrapper import PCWModelWrapper
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


def load_tokenizer(model_name: str, prompt_method: str=None) -> PreTrainedTokenizerBase:
    if 'llama' in model_name.lower() or 'gemma' in model_name.lower() \
        or 'qwen' in model_name.lower():
        if model_name == 'seanmor5/tiny-llama-test' or 'decapoda-research' in model_name:  # debug mode:
            tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
            # In case you load those models, we must override an incorrect config:
            # see: https://huggingface.co/decapoda-research/llama-7b-hf/discussions/12
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
        else:
            if prompt_method =="complex_cot":
               #assert 1==0
                tokenizer = AutoTokenizer.from_pretrained(model_name) #LlamaTokenizer.from_pretrained(model_name)
            else:
                # tokenizer = LlamaTokenizer.from_pretrained(model_name)
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
                     prompt_method: str = None, model_class: str = None, accelerator=None, 
                     Truncation_Method=None,parallel_pattern=None,attn_implementation=None,
                     raw_model_max_len=None,special_token=None,
                     key_no_rope=False,draw_pic=False,
                     attn_avg=False,in_eager_mode=False,
                     delete_context_prompt=False,context_prompt=None,
                     # attention calibration
                     input_stitching=False,calibration_mode=0,calibration_stage=None,
                     # kv cache parameters
                     kv_cache_eviction=False,kv_cache_dynamic=False,stage_eviction=False,capacity=None, 
                     # windows parameters
                     del_val=False,rank_windows=False,topk_windows=None,rank_cache=False,dynamic_window=False,
                     window_ascend=False,query_rank=False,decomposition_factor=1,
                     get_recent_attn=False,recent_top_num=0,query_recent_tokens=0,
                     # position shift parameters
                     position_shift=False,shift_factor=2,interval_shift=0,
                     # positional sorting
                     positional_sorting=None,rank_ascend=False,
                     # other try
                     parallel_decoding=False,window_pe_shift=False,NTK_aware=False,right_indentation: bool = False,
                     ) -> PCWModelWrapper:
    print("rank_windows:{}".format(rank_windows))
    print("model_name:{}".format(model_name))
#    assert 1==0
    validate_model_name(model_name)
    config = AutoConfig.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpus = torch.cuda.device_count() > 1
    model_args = {
        "cache_dir": cache_dir
    }
    print("multi_gpus:{}".format(multi_gpus))
    print("torch.cuda.device_count():{}".format(torch.cuda.device_count()))


    


    if 'gpt2' in model_name:
        # we override n_positions to bi pass the model's context window size restriction
        # (for gpt2, n_positions determines the causal attention mask matrix dimension).
        # The correct position embeddings (i.e., gpt2's 1024 trained position embeddings) are re-inserted to the model
        # in GPT2LMHeadWithPCWModel initialization.
        model_args['ignore_mismatched_sizes'] = True
        model_args['n_positions'] = GPT2_WINDOW_SIZE * n_windows
        model_obj = GPT2LMHeadPCW
        context_window_size = GPT2_WINDOW_SIZE
    else:
        #  Note that some LLaMa versions located in HF have an incorrect token mapping, we correct it here:
        # see: https://huggingface.co/decapoda-research/llama-7b-hf/discussions/12
        # also: https://github.com/tloen/alpaca-lora/issues/279
        
        if prompt_method == "complex_cot":
            model_obj = AutoModelForCausalLM#LlamaForCausalLM
            model = model_obj.from_pretrained(model_name, **model_args).eval()
        elif prompt_method == "complex_cot_pcw" or prompt_method == "complex_cot_pcw_multi_windows" \
            or prompt_method == "complex_cot_pcw_pre_process_window_cache"\
                or prompt_method == "complex_cot_pcw_multi_windows_kv_cache":
            if model_class == "modeling_llama_with_pcw":
               from modeling_llama_with_pcw import LlamaForCausalLMPCW
               model_obj = LlamaForCausalLMPCW
            elif model_class == "modeling_llama_with_pcw_wo_max_pos":
               from modeling_llama_with_pcw_wo_max_pos import LlamaForCausalLMPCW
               model_obj = LlamaForCausalLMPCW
            elif model_class == "modeling_llama_with_pcw_kv_cache":
                logger.info("modeling_llama_with_pcw_kv_cache is used")
                from modeling_llama_with_pcw_kv_cache import LlamaForCausalLMPCW
                model_obj = LlamaForCausalLMPCW
            elif model_class == "modeling_llama_with_pcw_kv_cache_longbench":
                logger.info("modeling_llama_with_pcw_kv_cache_longbench is used")
                from modeling_llama_with_pcw_kv_cache_longbench import LlamaForCausalLMPCW
                model_obj = LlamaForCausalLMPCW
            elif  model_class == "modeling_llama_with_pcw_kv_cache_FlashAttention_longbench":
                from modeling_llama_with_pcw_kv_cache_FlashAttention_longbench import LlamaForCausalLMPCW
                model_obj = LlamaForCausalLMPCW
            elif  model_class == "modeling_gemma_with_pcw_kv_cache_FlashAttention_longbench":
                from modeling_gemma_with_pcw_kv_cache_FlashAttention_longbench import Gemma2ForCausalLMPCW
                model_obj = Gemma2ForCausalLMPCW
            elif model_class == "modeling_qwen2_with_pcw_kv_cache_FlashAttention_longbench":
                from modeling_qwen2_with_pcw_kv_cache_FlashAttention_longbench import Qwen2ForCausalLMPCW
                model_obj = Qwen2ForCausalLMPCW
            elif  model_class == "modeling_llama_with_pcw_kv_cache_FlashAttention_longbench_437":
                from modeling_llama_with_pcw_kv_cache_FlashAttention_longbench_437 import LlamaForCausalLMPCW
                model_obj = LlamaForCausalLMPCW
            elif model_class == "modeling_llama_with_pcw_kv_cache_longbench_avg_logit":
                logger.info("modeling_llama_with_pcw_kv_cache_longbench_avg_logit is used")
                from modeling_llama_with_pcw_kv_cache_longbench_avg_logit import LlamaForCausalLMPCW
                model_obj = LlamaForCausalLMPCW
            elif model_class == "modeling_llama":
               from transformers.models.llama.modeling_llama import LlamaForCausalLM
               model_obj = LlamaForCausalLM
            #print("accelerator.process_index:{}".format(accelerator.process_index))
            
            if "kv_cache" in model_class:
               
#                print("model_obj:{}".format(model_obj))
# #               assert 1==0
#                 if capacity==1122:
#                     attn_implementation = "flash_attention_2"
#                 else:
#                     attn_implementation = 'eager'

                old_init = LlamaRotaryEmbedding.__init__
                # nums = 0
                def ntk_scaled_init(
                        self,
                        dim=None,
                        max_position_embeddings=2048,
                        base=10000,
                        device=None,
                        scaling_factor=1.0,
                        rope_type="default",
                        config: Optional[LlamaConfig] = None,
                    ):
                    # nonlocal nums
                    # nums+=1
                    # 修改config 只需要修改一次
                    if not config.revise:
                        # new max_position_embeddings and base
                        # max_position_embeddings = 16384
                        max_position_embeddings = 128000
                        base = config.rope_theta
                        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
                        dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)
                        
                        a = 8 #Alpha value
                        base = base * a ** (dim / (dim-2)) #Base change formula
                        config.rope_theta = base
                        # raw_max_position_embeddings = config.max_position_embeddings
                        config.max_position_embeddings = max_position_embeddings
                        config.revise = True
                    # 进入原始的init
                    old_init(self, dim, max_position_embeddings, base, device, scaling_factor, rope_type, config)
                if NTK_aware :
                    LlamaRotaryEmbedding.__init__ = ntk_scaled_init
                print(f"key_no_rope : {key_no_rope}")
                model = model_obj.from_pretrained(model_name, capacity = capacity, 
                                                 n_windows=n_windows, 
                                                 kv_cache_eviction=kv_cache_eviction,
                                                 kv_cache_dynamic=kv_cache_dynamic,
                                                 get_recent_attn=get_recent_attn,
                                                 stage_eviction=stage_eviction,
                                                 draw_pic=draw_pic,
                                                 calibration_mode=calibration_mode,
                                                 calibration_stage=calibration_stage,
                                                 in_eager_mode=in_eager_mode,
                                                 parallel_pattern=parallel_pattern,
                                                 key_no_rope=key_no_rope,
                                                 attn_avg=attn_avg,
                                                 attn_implementation=attn_implementation,
                                                 torch_dtype=torch.float16,
                                                 device_map={"": accelerator.process_index},
                                                #  device_map = "auto",
                                                 **model_args).eval()
                # model = dispatch_model(model, device_map="auto")
                # print(f"CUDA 是否可用: {torch.cuda.is_available()}")
                # model = accelerator.prepare(model)
                
                model.half()
                # assert 1==0
            else:
                model = model_obj.from_pretrained(model_name,**model_args).eval()
                model.half()
            # if multi_gpus:
            #    model = accelerator.prepare(model)
            
            if hasattr(model, "module"):
               model = model.module
        else:
                
            if hasattr(config, "torch_dtype") and config.torch_dtype is not None:   # 这里的代码导致生成不正常!!!!!!!!!!!!!!!!
                model_args["torch_dtype"] = config.torch_dtype
            model_args['bos_token_id'] = 1
            model_args['eos_token_id'] = 2
            model_obj = LlamaForCausalLMPCW
            model = model_obj.from_pretrained(model_name, **model_args).eval()
        context_window_size = LLAMA_WINDOW_SIZE
    
    tokenizer = load_tokenizer(model_name, prompt_method=prompt_method)
    # 1226
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 给model添加属性
    # if not multi_gpus:
    #     model = model.to(device)
    
    # if "anchor" in parallel_pattern:
    #     special_tokens_dict = {
    #         "additional_special_tokens": ["[Anchor]"]
    #     }
    #     tokenizer.add_special_tokens(special_tokens_dict)
    #     print(f" [Anchor] token id is {tokenizer.convert_tokens_to_ids('[Anchor]')}")  
    #     model.resize_token_embeddings(len(tokenizer))

    return PCWModelWrapper(model, tokenizer, device, context_window_size,
                           # base parameters
                           prompt_method=prompt_method, n_windows=n_windows, 
                           model_name=model_name,
                           parallel_pattern=parallel_pattern,Truncation_Method=Truncation_Method,
                           parallel_decoding=parallel_decoding,
                           raw_model_max_len=raw_model_max_len,special_token=special_token,
                           delete_context_prompt=delete_context_prompt,context_prompt=context_prompt,
                           # kv_cache parameters
                           capacity=capacity, kv_cache_eviction=kv_cache_eviction, stage_eviction=stage_eviction,
                           # windows parameters
                           del_val=del_val, rank_windows=rank_windows, topk_windows=topk_windows, 
                           rank_cache=rank_cache,dynamic_window=dynamic_window,
                           window_ascend=window_ascend,query_rank=query_rank,
                           decomposition_factor=decomposition_factor,
                           get_recent_attn=get_recent_attn,recent_top_num=recent_top_num,
                           query_recent_tokens=query_recent_tokens,
                           # position shift parameters
                           position_shift=position_shift,shift_factor=shift_factor,
                           interval_shift=interval_shift,
                           # positional sorting
                           positional_sorting=positional_sorting,rank_ascend=rank_ascend,
                           # other try
                           NTK_aware=NTK_aware,right_indentation=right_indentation,
                           # attention calibration
                           input_stitching=input_stitching,calibration_stage=calibration_stage,calibration_mode=calibration_mode,
                           )
