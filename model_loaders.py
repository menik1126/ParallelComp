import torch
from transformers import AutoConfig, LlamaTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM

from accelerate import Accelerator
from modeling_gpt2_with_pcw import GPT2LMHeadPCW
from pcw_wrapper import PCWModelWrapper

from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch
# accelerator = Accelerator()
GPT2_WINDOW_SIZE = 1024
LLAMA_WINDOW_SIZE = 2048


def validate_model_name(model_name: str) -> None:
    assert 'llama' in model_name or 'gpt2' in model_name, f"Unknown model: {model_name}"


def load_tokenizer(model_name: str, prompt_method: str=None) -> PreTrainedTokenizerBase:
    if 'llama' in model_name:
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
               tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        # In our experiments we have added bos token to gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token=True)
    return tokenizer


def load_pcw_wrapper(model_name: str, cache_dir: str = None,
                     right_indentation: bool = False, n_windows: int = 1, prompt_method: str = None, model_class: str = None, accelerator=None) -> PCWModelWrapper:
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

    # if multi_gpus:
        
    #     model_args["device_map"] = {"": accelerator.process_index}
            #model_args["low_cpu_mem_usage"] = True
    


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
               from modeling_llama_with_pcw_kv_cache import LlamaForCausalLMPCW
               model_obj = LlamaForCausalLMPCW
            elif model_class == "modeling_llama":
               from transformers.models.llama.modeling_llama import LlamaForCausalLM
               model_obj = LlamaForCausalLM
            #print("accelerator.process_index:{}".format(accelerator.process_index))
            model = model_obj.from_pretrained(model_name, **model_args).eval()
            if multi_gpus:
               

               model = accelerator.prepare(model)
            
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
    
    if not multi_gpus:
        model = model.to(device)

    # 给model添加属性
    

    return PCWModelWrapper(model, tokenizer, device, context_window_size, right_indentation, prompt_method=prompt_method, n_windows=n_windows)
