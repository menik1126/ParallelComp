import math
from abc import ABC
from typing import Optional, Tuple, Dict, Union, List
import torch.nn.functional as F
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm,apply_rotary_pos_emb, \
    LlamaDecoderLayer, LlamaModel, LlamaForCausalLM, repeat_kv, LlamaFlashAttention2, StaticCache, LlamaRotaryEmbedding, \
        LlamaMLP
from transformers.cache_utils import Cache,DynamicCache
# from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal
from pcw_wrapper import generate_pcw_position_ids
from transformers.modeling_flash_attention_utils import _upad_input, flash_attn_varlen_func, flash_attn_func
from my_utils.augment_index import aug_indices_comb_classification
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
import os
import copy
import inspect
import logging
from torch.nn import CrossEntropyLoss
from my_utils.logger import Logger
logger = Logger()
logger.set_console_level(logging.DEBUG)

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1",
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
        causal = is_causal and query_length != 1
    
    
    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if is_flash_attn_greater_or_equal("2.4.1"):
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            # return_attn_probs = True,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, 
            # return_attn_probs = True,
            **flash_kwargs
        )
        
    return attn_output
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_new(q, k, cos_query, sin_query, cos_key,sin_key, position_ids=None, unsqueeze_dim=1):
    cos_query = cos_query.unsqueeze(unsqueeze_dim)
    sin_query = sin_query.unsqueeze(unsqueeze_dim)
    cos_key = cos_key.unsqueeze(unsqueeze_dim)
    sin_key = sin_key.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_query) + (rotate_half(q) * sin_query)
    k_embed = (k * cos_key) + (rotate_half(k) * sin_key)
    return q_embed, k_embed

HEAD_NUM = int(os.environ.get("HEAD_NUM", 0))
BETA = float(os.environ.get("BETA", 0.4))
THRES = float(os.environ.get("THRES", 0.1))
RECENT_TOKENS = int(os.environ.get("RECENT_TOKENS", 1))


class LlamaFlashAttention2PCW(LlamaFlashAttention2):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None,
                capacity = 512, n_windows=1, kv_cache_eviction=False,
                kv_cache_dynamic=False,stage_eviction=False,get_recent_attn=False,
                key_no_rope=False,draw_pic=0,attn_avg=False,
                in_eager_mode=False,
                calibration_mode=0,calibration_stage=None,
                 ):
        super().__init__(config, layer_idx)
        self.capacity = capacity
        self.eviction_len = 0
        self.window_size = 8
        self.kv_cache_eviction_prefill = kv_cache_eviction
        self.kv_cache_dynamic = kv_cache_dynamic
        self.query_len = self.window_size
        self.n_windows = n_windows
        self.stage_eviction = stage_eviction
        self.key_no_rope = key_no_rope
        self.attn_avg = attn_avg
        self.in_eager_mode = in_eager_mode
        self.kernel_size = 7
        self.get_recent_attn = get_recent_attn
        self.pooling = 'maxpool'
        self.draw_pic = draw_pic
        self.calibration_mode = calibration_mode
        self.calibration_stage=calibration_stage
    
    def atten_process_eval(self, attention_map, index=None):
        # 核心思想为：把sink位置的注意力分数，均摊1-beta部分到其他位置（按照本身自己的权重分配）
        beta = BETA
        threshold = THRES

        head_indices = aug_indices_comb_classification[index-2]

        for head_num in head_indices:
            modified_head = attention_map[0][head_num]
            device = modified_head.device
            shape = modified_head.shape[1]
            # print("torch.sum(modified_head, dim=0).shape:{}".format(torch.sum(modified_head, dim=0).shape))
            # print("torch.arange(shape,0,-1).to(device).shape:{}".format(torch.arange(shape,0,-1).to(device).shape))
            
            # 计算平均注意力大于阈值的部分（寻找注意力sink的位置）
            indices = torch.nonzero(torch.where(torch.sum(modified_head, dim=0) / torch.arange(shape,0,-1).to(device) > threshold, 1, 0))
            # print("indices.shape:{}".format(indices.shape))
            indices = indices[1:]
            copied_attention_map = copy.deepcopy(modified_head.detach())
            # 获得权重 sink位置作和*一个系数
            available_weights = modified_head[:, indices].sum(dim=1) * (1-beta)
            # 对sink部分进行缩放
            modified_head[:, indices] *= beta
            # 掩盖掉 sink部分
            copied_attention_map[1:, indices] *= 0
            # 计算剩余部分的权重
            ratios = copied_attention_map / torch.sum(copied_attention_map,  dim=1, keepdim=True).to(copied_attention_map.dtype)
            # 对available_weights进一步缩放
            modified_head = modified_head + available_weights * ratios
            modified_head[0, 0] = 1
           
            attention_map[0][head_num] = modified_head

        return attention_map
    
    def atten_process_eval_generate(self, attention_map, index=None):
        # 核心思想为：把sink位置的注意力分数，均摊1-beta部分到其他位置（按照本身自己的权重分配）
        beta = BETA
        threshold = THRES

        head_indices = aug_indices_comb_classification[index-2]

        for head_num in head_indices:
            # print(f"attention_map before:{attention_map}")
            # print(f"\ntorch.sum(attention_map):{torch.sum(attention_map)}")
            # print("attention_map.shape:{}".format(attention_map.shape))
            modified_head = attention_map[0][head_num]
            device = modified_head.device
            shape = modified_head.shape[1]
            # print("torch.sum(modified_head, dim=0).shape:{}".format(torch.sum(modified_head, dim=0).shape))
            # print("torch.arange(shape,0,-1).to(device).shape:{}".format(torch.arange(shape,0,-1).to(device).shape))
            
            # 计算平均注意力大于阈值的部分（寻找注意力sink的位置）
            indices = torch.nonzero(torch.where(torch.sum(modified_head, dim=0) / torch.arange(shape,0,-1).to(device) > threshold, 1, 0))
            indices = indices[1:]
            copied_attention_map = copy.deepcopy(modified_head.detach())
            # 获得权重 sink位置作和*一个系数
            available_weights = modified_head[:, indices].sum(dim=1) * (1-beta)
            # 对sink部分进行缩放
            modified_head[:, indices] *= beta
            # 掩盖掉 sink部分
            copied_attention_map[1:, indices] *= 0
            # 计算剩余部分的权重
            ratios = copied_attention_map / torch.sum(copied_attention_map,  dim=1, keepdim=True).to(copied_attention_map.dtype)
            # 对available_weights进一步缩放
            modified_head = modified_head + available_weights * ratios
            # modified_head[0, 0] = 1
           
            attention_map[0][head_num] = modified_head
            # print(f"attention_map:{attention_map}")
            # print(f"torch.sum(attention_map):{torch.sum(attention_map)}")
            # print(f"torch.sum(attention_map).shape:{torch.sum(attention_map).shape}")
            # assert 1==0
        return attention_map
    
    def atten_process_eval_prefill(self, attention_map, layer_idx=None, threshold=0.1, beta=0.4):
        # 核心思想为：把sink位置的注意力分数，均摊1-beta部分到其他位置（按照本身自己的权重分配）
        
        # 每个头都操作
        for head_num in range(attention_map.shape[1]):
            modified_head = attention_map[0][head_num]
            device = modified_head.device
            shape = modified_head.shape[1]
            # attn_comparison = torch.sum(modified_head, dim=0) / torch.arange(shape,0,-1).to(device)
            
            # 计算平均注意力大于阈值的部分（寻找注意力sink的位置）
            self.window_size = min(self.window_size, modified_head.shape[0])
            attn_comparison = torch.sum(modified_head[-self.window_size:,:], dim=0)
            attn_comparison[:-self.window_size+1] /= self.window_size
            for i in range(1,self.window_size):
                attn_comparison[-i] /= i
            indices = torch.nonzero(torch.where(attn_comparison > threshold, 1, 0))
            copied_attention_map = copy.deepcopy(modified_head.detach())
            # 获得权重 sink位置作和*一个系数
            available_weights = modified_head[:, indices].sum(dim=1) * (1-beta)
            # 对sink部分进行缩放
            modified_head[:, indices] *= beta
            # 掩盖掉 sink部分
            copied_attention_map[:, indices] *= 0
            # 计算剩余部分的权重
            attn_map_sum = torch.sum(copied_attention_map,  dim=1, keepdim=True)
            small_value= 1e-8
            attn_map_sum = torch.where(attn_map_sum == 0, small_value, attn_map_sum)
            ratios = copied_attention_map / attn_map_sum.to(copied_attention_map.dtype)
            modified_head = modified_head + available_weights * ratios
            if attention_map.shape[-1] == attention_map.shape[-2]:
                modified_head[0, 0] = 1
            
        return attention_map
    
    def atten_process_eval_generate_new(self, attention_map, layer_idx=None, threshold=0.1, beta=0.4):
        # 核心思想为：把sink位置的注意力分数，均摊1-beta部分到其他位置（按照本身自己的权重分配）
        # 每个头都操作
        # 0.3 0.3还可以
        
        for head_num in range(attention_map.shape[1]):
            modified_head = attention_map[0][head_num]
            device = modified_head.device
            shape = modified_head.shape[1]
            # attn_comparison = torch.sum(modified_head, dim=0) / torch.arange(shape,0,-1).to(device)
            attn_comparison = modified_head[0]
            # 计算平均注意力大于阈值的部分（寻找注意力sink的位置）
            indices = torch.nonzero(torch.where( attn_comparison > threshold, 1, 0))
            # print("indices.shape:{}".format(indices.shape))
            # indices = indices[1:]
            copied_attention_map = copy.deepcopy(modified_head.detach())
            # 获得权重 sink位置作和*一个系数
            available_weights = modified_head[:, indices].sum(dim=1) * (1-beta)
            # 对sink部分进行缩放
            modified_head[:, indices] *= beta
            # 掩盖掉 sink部分
            copied_attention_map[:, indices] *= 0
            # 计算剩余部分的权重
            ratios = copied_attention_map / torch.sum(copied_attention_map,  dim=1, keepdim=True).to(copied_attention_map.dtype)
            # 对available_weights进一步缩放
            modified_head = modified_head + available_weights * ratios
           
            attention_map[0][head_num] = modified_head
        return attention_map

    def _update_kv_second_prefill(self,past_key_value, key_states, query_states, value_states, q_len):
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
        self.query_len = self.window_size
        # print("attn_weights_sum.shape:{}".format(attn_weights_sum.shape))
        # assert 1==0
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        #print("self.capacity0:{}".format(self.capacity))
        #assert 1==0
        capacity = min(self.capacity, q_len)
        top_k = capacity*self.n_windows-self.query_len
        # logger.debug(f"top_k:{top_k}")
        # logger.debug(f"capacity:{capacity}")
        # logger.debug(f"self.query_len:{self.query_len}")
        # logger.debug(f"attn_weights_sum.shape: {attn_weights_sum.shape}")
        self.new_capacity = capacity*self.n_windows
        indices = attn_cache.topk(top_k, dim=-1).indices
        indices = indices.sort(dim=-1).values
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = attn_weights[:, :, -self.window_size:, :-self.query_len].gather(dim = -1, index = indices_attn)
        k_past_compress = key_states[:, :, :-self.query_len, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, :-self.query_len, :].gather(dim = 2, index = indices_expanded)
        k_cur = key_states[:, :, -self.query_len:, :]
        v_cur = value_states[:, :, -self.query_len:, :]
        attn_weights_cur = attn_weights[:, :, -self.window_size:, -self.query_len:]
        
        key_states_compress = torch.cat([k_past_compress, k_cur], dim = 2).contiguous()
        value_states_compress = torch.cat([v_past_compress, v_cur], dim = 2).contiguous()
        self.attn_weights = torch.cat([attn_weights_compress,attn_weights_cur],dim=-1)
        # logger.debug(f"self.attn_weights.shape:{self.attn_weights.shape}")
        
        del k_past_compress, k_cur, v_past_compress, v_cur, key_states, value_states, attn_weights_compress, attn_weights_cur
        torch.cuda.empty_cache()
        
        past_key_value.key_cache[self.layer_idx] = key_states_compress
        past_key_value.value_cache[self.layer_idx] = value_states_compress

    def _update_kv_first_prefill(self,past_key_value, key_states, query_states, value_states, q_len):
        token_prompt_size = self.token_prompt_size
        
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, token_prompt_size : -self.query_len].sum(dim = -2)
        # print("attn_weights_sum.shape:{}".format(attn_weights_sum.shape))
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        
        #print("self.capacity0:{}".format(self.capacity))
        #assert 1==0
        capacity = min(self.capacity, q_len)
        # logger.info("capacity:{}".format(capacity))
        top_k = capacity-self.query_len
        # logger.info("top_k:{}".format(top_k))
        # logger.info("attn_cache.shape:{}".format(attn_cache.shape))
        indices = attn_cache.topk(top_k, dim=-1).indices
        indices = indices.sort(dim=-1).values
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = attn_weights[:, :, -self.window_size:, token_prompt_size:-self.query_len].gather(dim = -1, index = indices_attn)
        k_past_compress = key_states[:, :, token_prompt_size:-self.query_len, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, token_prompt_size:-self.query_len, :].gather(dim = 2, index = indices_expanded)
        k_cur = key_states[:, :, -self.query_len:, :]
        v_cur = value_states[:, :, -self.query_len:, :]

        key_states_compress = torch.cat([k_past_compress, k_cur], dim = 2).contiguous()
        value_states_compress = torch.cat([v_past_compress, v_cur], dim = 2).contiguous()

        del k_past_compress, k_cur, v_past_compress, v_cur, key_states, value_states, attn_weights_compress
        torch.cuda.empty_cache()
        
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx,cache_kwargs=None)
    
    def _update_kv_first_prefill_batches(self,past_key_value, key_states, query_states, value_states, q_len):
        token_prompt_size = self.token_prompt_size
        
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # print("attn_weights.shape",attn_weights.shape)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, token_prompt_size : -self.query_len].sum(dim = -2)
        # print("attn_weights_sum.shape:{}".format(attn_weights_sum.shape))
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        
        #print("self.capacity0:{}".format(self.capacity))
        #assert 1==0
        capacity = min(self.capacity, q_len)
        # logger.info("capacity:{}".format(capacity))
        top_k = capacity-self.query_len
        # logger.info("top_k:{}".format(top_k))
        # logger.info("attn_cache.shape:{}".format(attn_cache.shape))
        indices = attn_cache.topk(top_k, dim=-1).indices
        indices = indices.sort(dim=-1).values
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = attn_weights[:, :, -self.window_size:, token_prompt_size:-self.query_len].gather(dim = -1, index = indices_attn)
        k_past_compress = key_states[:, :, token_prompt_size:-self.query_len, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, token_prompt_size:-self.query_len, :].gather(dim = 2, index = indices_expanded)
        k_cur = key_states[:, :, -self.query_len:, :]
        v_cur = value_states[:, :, -self.query_len:, :]

        key_states_compress = torch.cat([k_past_compress, k_cur], dim = 2).contiguous()
        value_states_compress = torch.cat([v_past_compress, v_cur], dim = 2).contiguous()

        del k_past_compress, k_cur, v_past_compress, v_cur, key_states, value_states, attn_weights_compress
        torch.cuda.empty_cache()
        
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx,cache_kwargs=None)
    
              
    def _update_kv_first_prefill_key_no_rope(self,past_key_value, key_states,before_rope_key_states,query_states, value_states, q_len):
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.query_len].sum(dim = -2)
        # print("attn_weights_sum.shape:{}".format(attn_weights_sum.shape))
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        
        #print("self.capacity0:{}".format(self.capacity))
        #assert 1==0
        capacity = min(self.capacity, q_len)
        # logger.info("capacity:{}".format(capacity))
        top_k = capacity-self.query_len
        indices = attn_cache.topk(top_k, dim=-1).indices
        indices = indices.sort(dim=-1).values
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = attn_weights[:, :, -self.window_size:, :-self.query_len].gather(dim = -1, index = indices_attn)
        k_past_compress = before_rope_key_states[:, :, :-self.query_len, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, :-self.query_len, :].gather(dim = 2, index = indices_expanded)
        k_cur = before_rope_key_states[:, :, -self.query_len:, :]
        v_cur = value_states[:, :, -self.query_len:, :]

        key_states_compress = torch.cat([k_past_compress, k_cur], dim = 2).contiguous()
        value_states_compress = torch.cat([v_past_compress, v_cur], dim = 2).contiguous()

        del k_past_compress, k_cur, v_past_compress, v_cur, key_states, value_states, attn_weights_compress
        torch.cuda.empty_cache()
        
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx,cache_kwargs=None)
        
    def _update_kv_input(self, attn_weights, kv_seq_len):
        # 更新capacity的长度为 拼接后的+input
        self.new_capacity = attn_weights.shape[-1]
        # 重新确定window_size
        self.window_size = min(self.window_size,kv_seq_len)
        self.attn_weights = attn_weights[:,:,-self.window_size:,:]
        
    def _update_kv_generate(self, attn_weights, past_key_value,key_states,value_states,layer_idx):
        _,_,_,head_dim = key_states.shape
        attn_weights_dynamic = self.attn_weights
        # logger.debug(f"attn_weights.shape:{attn_weights.shape}") 
        now_attn_weights = torch.zeros((attn_weights_dynamic.shape[0],attn_weights_dynamic.shape[1],attn_weights_dynamic.shape[2]+1,attn_weights_dynamic.shape[3]+1),device=attn_weights_dynamic.device)
        now_attn_weights[:,:,:-1,:-1] = attn_weights_dynamic
        now_attn_weights[:,:,:,-1] = 0
        now_attn_weights[:,:,-1:,:] = attn_weights
        attn_weights_sum = now_attn_weights[...,-self.window_size:,:-self.window_size].sum(-2)
        cache_size = self.new_capacity
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        indices  = attn_cache.topk(cache_size-self.window_size, dim=-1).indices
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = now_attn_weights[:, :, -self.window_size:, :-self.window_size].gather(dim = -1, index = indices_attn)
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices_expanded)
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        attn_weights_cur = now_attn_weights[:, :, -self.window_size:, -self.window_size:]
        revise_key_states = torch.cat([k_past_compress, k_cur], dim = 2)
        revise_value_states = torch.cat([v_past_compress, v_cur], dim = 2)
        self.attn_weights = torch.cat([attn_weights_compress,attn_weights_cur],dim=-1)
        
        past_key_value.key_cache[layer_idx] = revise_key_states
        past_key_value.value_cache[layer_idx] = revise_value_states
    
    def get_attn(self,query_states, key_states):
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :].sum(dim = -2)
        # print("attn_weights_sum.shape:{}".format(attn_weights_sum.shape))
        # if self.pooling == 'avgpool':
        #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        # elif self.pooling == 'maxpool':
        #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        # else:
        #     raise ValueError('Pooling method not supported')

        return attn_weights_sum
    
    def draw_attn(self,query_states, key_states):
        if self.window_size == 0:
            self.window_size = query_states.shape[-2]
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights_sum = attn_weights[:, :, -self.window_size:, :].sum(dim = -2)

        return attn_weights
    
    def eviction_tokens(self,key_states, value_states, attn_weights,layer_idx,threshold, recent_tokens,calibration_stage):
        parts = calibration_stage.split('_')
        last_two_numbers = list(map(int, parts[-2:]))
        # print("last_two_numbers:{}".format(last_two_numbers))
        if last_two_numbers[0] <= layer_idx <= last_two_numbers[1]:
            attn_sum = attn_weights[:,:,-recent_tokens:,:].sum(0).sum(0).sum(0) / (attn_weights.shape[1]*recent_tokens)
            if torch.sum(attn_weights[:,:,-recent_tokens:,:]) != 32*recent_tokens:
                print(f"torch.sum(attn_sum) : {torch.sum(attn_weights[:,:,-recent_tokens:,:])}")
                raise ValueError("Attention sum is not equal to 32*recent_tokens")
            shape = attn_sum.shape[0]
            device = attn_sum.device
            indices = torch.nonzero(torch.where( attn_sum > threshold, 1, 0))
            indices = indices.squeeze(-1)
            reserved_indices = torch.arange(shape, device=device)
            indices = torch.tensor(list(set(reserved_indices.tolist())-set(indices.tolist())),device=device)
            
            new_key_states = key_states[:,:,indices,:]
            new_value_states = value_states[:,:,indices,:]
        else:
            new_key_states = key_states
            new_value_states = value_states
        return new_key_states, new_value_states
    
    def eviction_tokens_head(self,key_states, value_states, attn_weights,layer_idx,threshold, recent_tokens,calibration_stage,pattern):
        parts = calibration_stage.split('_')
        last_two_numbers = list(map(int, parts[-2:]))
        if last_two_numbers[0] <= layer_idx <= last_two_numbers[1]:
            if recent_tokens == 0:
                recent_tokens = attn_weights.shape[-2]
            attn_sum = attn_weights[:,:,-recent_tokens:,:].sum(0).sum(1) / (attn_weights.shape[0]*recent_tokens)
            all_indices = []
            for i in range(attn_sum.shape[0]):
                indices = torch.nonzero(torch.where( attn_sum[i] > threshold, 1, 0)).squeeze(-1)
                # 小于100的部分进行驱逐
                if pattern == "recent":  # 驱逐recent,保留sink
                    filtered_indices = indices[indices > attn_sum.shape[1]-200]
                elif pattern == "sink": # 驱逐sink,保留recent
                    filtered_indices = indices[indices < 200]
                elif pattern =="middle" : # 驱逐middel
                    indices_temp = indices[indices < attn_sum.shape[1]-200] 
                    filtered_indices = indices_temp[ indices_temp > 200]
                elif pattern == "all": # 全部驱逐
                    filtered_indices = indices
                elif pattern == "recent_sink":
                    indices_recent = indices[indices > attn_sum.shape[1]-200] 
                    indices_sink = indices[ indices < 200]
                    filtered_indices = torch.cat([indices_recent,indices_sink],dim=-1)
                all_indices.append(filtered_indices)
            len_indices = [indices.shape[0] for indices in all_indices]
            max_len = max(len_indices)
            len_supplement_indices = [max_len - len_index for len_index in len_indices]
            # 给每个head补充的部分
            for i in range(attn_sum.shape[0]):
                attn_now = attn_sum[i] 
                # N长度的token
                supplement_indices = attn_now.topk(len_supplement_indices[i], dim=-1,largest=False ).indices
                all_indices[i] = torch.cat([all_indices[i],supplement_indices],dim=-1)
            all_indices = torch.stack(all_indices)
            full_indices = torch.arange(key_states.shape[2], device=key_states.device).unsqueeze(0).expand(attn_sum.shape[0],-1)
            mask = torch.ones_like(full_indices)
            mask.scatter_(1, all_indices, 0)
            mask_bool = mask.bool()
            select_indices = full_indices[mask_bool].reshape(attn_sum.shape[0],-1)
            new_key_states = key_states.gather(dim=2,index=select_indices.unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,key_states.shape[-1]))
            new_value_states = value_states.gather(dim=2,index=select_indices.unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,value_states.shape[-1]))
        else:
            new_key_states = key_states
            new_value_states = value_states
        return new_key_states, new_value_states
    
    
    def eviction_tokens_first_prefill_head(self,key_states, value_states, attn_weights,layer_idx,threshold, recent_tokens,calibration_stage,pattern):
        token_prompt_size = self.token_prompt_size
        query_len = self.query_len
        # 不关注query和开始符号部分（保留下来）
        parts = calibration_stage.split('_')
        last_two_numbers = list(map(int, parts[-2:]))
        if last_two_numbers[0] <= layer_idx <= last_two_numbers[1]:
            if recent_tokens == 0:
                recent_tokens = attn_weights.shape[-2]
            attn_sum = attn_weights[:,:,-recent_tokens:,token_prompt_size:-query_len].sum(0).sum(1) / (attn_weights.shape[0]*recent_tokens)
            all_indices = []
            for i in range(attn_sum.shape[0]):
                indices = torch.nonzero(torch.where( attn_sum[i] > threshold, 1, 0)).squeeze(-1)
                indices += token_prompt_size
                # 小于100的部分进行驱逐
                if pattern == "recent":  # 驱逐recent,保留sink
                    filtered_indices = indices[indices > attn_sum.shape[1]-200]
                elif pattern == "sink": # 驱逐sink,保留recent
                    filtered_indices = indices[indices < 200]
                elif pattern =="middle" : # 驱逐middel
                    indices_temp = indices[indices < attn_sum.shape[1]-200] 
                    filtered_indices = indices_temp[ indices_temp > 200]
                elif pattern == "all": # 全部驱逐
                    filtered_indices = indices
                all_indices.append(filtered_indices)
            len_indices = [indices.shape[0] for indices in all_indices]
            max_len = max(len_indices)
            
            self.eviction_len = max_len
            
            len_supplement_indices = [max_len - len_index for len_index in len_indices]
            # 给每个head补充的部分
            for i in range(attn_sum.shape[0]):
                attn_now = attn_sum[i] 
                # N长度的token
                supplement_indices = attn_now.topk(len_supplement_indices[i], dim=-1,largest=False ).indices
                supplement_indices += token_prompt_size
                all_indices[i] = torch.cat([all_indices[i],supplement_indices],dim=-1)
            all_indices = torch.stack(all_indices)
            full_indices = torch.arange(key_states.shape[2], device=key_states.device).unsqueeze(0).expand(attn_sum.shape[0],-1)
            mask = torch.ones_like(full_indices)
            mask.scatter_(1, all_indices, 0)
            mask_bool = mask.bool()
            select_indices = full_indices[mask_bool].reshape(attn_sum.shape[0],-1)
            new_key_states = key_states.gather(dim=2,index=select_indices.unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,key_states.shape[-1]))
            new_value_states = value_states.gather(dim=2,index=select_indices.unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,value_states.shape[-1]))
        else:
            new_key_states = key_states
            new_value_states = value_states
        return new_key_states, new_value_states
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )
        raw_len_past_key_value = len(past_key_value) 
        # print("raw_len_past_key_value:{}".format(raw_len_past_key_value))
        bsz, q_len, _ = hidden_states.size()
        output_attentions = False
        # 第一次prefill
        if raw_len_past_key_value!=self.config.num_hidden_layers:
            stage = 0 #"first_prefill"
        # 第二次prefill
        elif raw_len_past_key_value==self.config.num_hidden_layers and q_len != 1:
            stage = 1 #"second_prefill"
        # generate
        elif raw_len_past_key_value==self.config.num_hidden_layers and q_len == 1:
            stage = 2 #"generate"
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        before_rope_key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        if self.key_no_rope and  \
            raw_len_past_key_value==self.config.num_hidden_layers and q_len != 1: # 第二次prefill
            # 第一次更新，加载所有的key_states为添加位置编码前的key_states
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(before_rope_key_states, value_states, self.layer_idx, cache_kwargs)
            query_states, key_states = apply_rotary_pos_emb_new(query_states, key_states, cos[:,-q_len:,:], sin[:,-q_len:,:],cos,sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, before_rope_key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
          
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if raw_len_past_key_value == self.config.num_hidden_layers: # 第二次prefill和generate
                if self.key_no_rope and q_len != 1: # 第二次prefill
                    # 更新为添加了位置编码信息的key_states  
                    past_key_value.key_cache[self.layer_idx] = key_states
                else:
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            raw_query_states = query_states
            raw_key_states = key_states
            raw_value_states = value_states

        # 驱逐模式
        threshold = THRES
        beta= BETA
        # recent_tokens = self.window_size
        recent_tokens = RECENT_TOKENS
        if recent_tokens == 0:
            self.window_size = q_len
        # attn_avg
        if self.in_eager_mode:
            # print("haha")
            attn_weights = torch.matmul(query_states, raw_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            # 对注意力分数进行修正
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : raw_key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            # print("attention_mask[:,:,-50:,-50:]",attention_mask[:,:,-50:,-50:])
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            
            if self.calibration_mode == 1:
                #第一次prefill驱逐
                if self.calibration_stage == "prefill_2" and stage==1: 
                    raw_key_states,raw_value_states = self.eviction_tokens(raw_key_states,raw_value_states, 
                                                                           attn_weights, self.layer_idx,
                                                                           threshold, recent_tokens)
                    past_key_value.key_cache[self.layer_idx] = raw_key_states
                    past_key_value.value_cache[self.layer_idx] = raw_value_states
                elif self.calibration_stage == "prefill_1" and stage == 0:
                    raw_key_states,raw_value_states = self.eviction_tokens_first_prefill(raw_key_states,raw_value_states, 
                                                                           attn_weights, self.layer_idx,
                                                                           threshold, recent_tokens)
            elif self.calibration_mode == 2 or self.calibration_mode==22: #校准模式
                # 第一次preifll
                if self.attn_avg and stage == 1:
                    if stage == 0 or stage == 1:
                        ##############BEGIN:MODIFICATION##############
                        if 3 in range(2, 31):
                            attn_weights = self.atten_process_eval(attn_weights, index=2)
                        ##############END:MODIFICATION##############
                    elif stage == 2:
                        if 3 in range(2, 31):
                            attn_weights = self.atten_process_eval_generate(attn_weights, index=2)
                else:
                    if self.calibration_stage == "prefill_2" and stage == 1:
                        attn_weights = self.atten_process_eval_prefill(attn_weights,self.layer_idx,threshold,beta)
                    elif self.calibration_stage == "prefill_1" and stage == 0:
                        attn_weights = self.atten_process_eval_prefill(attn_weights,self.layer_idx,threshold,beta)
                    elif self.calibration_stage == "generate" and stage == 2:
                        attn_weights = self.atten_process_eval_generate_new(attn_weights,self.layer_idx,threshold,beta)
                    elif self.calibration_stage == "prefill_12" and (stage == 0 or stage ==1):
                        attn_weights = self.atten_process_eval_prefill(attn_weights,self.layer_idx,threshold,beta)
                    elif self.calibration_stage == "prefill_123" and (stage == 0 or stage ==1 or stage == 2):
                        if stage == 2:
                            attn_weights = self.atten_process_eval_generate_new(attn_weights,self.layer_idx,threshold,beta)
                        else:
                            attn_weights = self.atten_process_eval_prefill(attn_weights,self.layer_idx,threshold,beta)
                    
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, raw_value_states)
                attn_output = attn_output.transpose(1, 2)
            else:
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, raw_value_states)
                attn_output = attn_output.transpose(1, 2)
                pass
        else:
            # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            if self.calibration_mode != 0:
                attn_weights = torch.matmul(query_states[:,:,-recent_tokens:,:], raw_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, -recent_tokens:, : raw_key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            if self.calibration_mode == 1 or self.calibration_mode == 11 or self.calibration_mode == 111:
                #第二次prefill驱逐
                if "prefill_2" in self.calibration_stage and stage==1: 
                    if "head" in self.calibration_stage:
                        if "head1" in self.calibration_stage:
                            pattern = "recent"
                        elif "head2" in self.calibration_stage:
                            pattern = "sink"
                        elif "head3" in self.calibration_stage:
                            pattern = "middle"
                        elif "head4" in self.calibration_stage:
                            pattern = "all"
                        elif "head5" in self.calibration_stage:
                            pattern = "recent_sink"
                        raw_key_states,raw_value_states = self.eviction_tokens_head(raw_key_states,raw_value_states, 
                                                                           attn_weights, self.layer_idx,
                                                                           threshold, recent_tokens, self.calibration_stage,
                                                                           pattern = pattern)
                    else:
                        raw_key_states,raw_value_states = self.eviction_tokens(raw_key_states,raw_value_states, 
                                                                           attn_weights, self.layer_idx,
                                                                           threshold, recent_tokens, self.calibration_stage)                
                    # 更新past_key_value
                    # if "prefill_2_eviction" in self.calibration_stage:
                    # 保留校正后的key value（与attn一致）
                    past_key_value.key_cache[self.layer_idx] = raw_key_states
                    past_key_value.value_cache[self.layer_idx] = raw_value_states
                    # 校准，而不是驱逐token
                    if "prefill_2_calibration" in self.calibration_stage:
                        key_states = raw_key_states
                        value_states = raw_value_states
                        # print("I am in calibration")

                elif "prefill_1" in self.calibration_stage  and stage == 0:
                    if "head" in self.calibration_stage:
                        if "head1" in self.calibration_stage:
                            pattern = "recent"
                        elif "head2" in self.calibration_stage:
                            pattern = "sink"
                        elif "head3" in self.calibration_stage:
                            pattern = "middle"
                        elif "head4" in self.calibration_stage:
                            pattern = "all"
                            
                        raw_key_states,raw_value_states = self.eviction_tokens_first_prefill_head(raw_key_states,raw_value_states, 
                                                                           attn_weights, self.layer_idx,
                                                                           threshold, recent_tokens, self.calibration_stage,
                                                                           pattern = pattern)
                    # 只做矫正，不驱逐(仍然会影响第二次prefill和generate)
                    if "prefill_1_calibration" in self.calibration_stage:
                        key_states = raw_key_states
                        value_states = raw_value_states
                
                
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in the correct dtype just to be sure everything works as expected.
            # This might slowdown training & inference so it is recommended to not cast the LayerNorms
            # in fp32. (LlamaRMSNorm handles it correctly)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
            )
        
        if self.draw_pic!= 0:
            self.get_draw_attn = self.draw_attn(raw_query_states,raw_key_states)
            import json
            if self.draw_pic == 1:
                names = "first_prefill/datas"
            elif self.draw_pic == 2:
                names = "second_prefill/datas"
                names = "first_prefill_compression/datas"
                names = "first_prefill_new/datas"
            elif self.draw_pic == 3:
                names = "generate/datas"
            elif self.draw_pic == 4:
                names = "eviction_all/datas"
            elif self.draw_pic == 5:
                names = "new/generate_after_kv_compression/datas"
            #交换顺序
            pattern = "0327/narrativeqa/"
            # pattern = ""
            output_path = "/home/avnet/xiongjing/sjh/parallel_window/draw_picture/"+ pattern + names + "/"
            os.makedirs(output_path, exist_ok=True)
            output_datapath = os.path.join(output_path,
                                           f"layer_{self.layer_idx}.json"
                                           )
            # logger.info(f"output_datapath: {output_datapath}")
            # logger.info(f"self.get_draw_attn.shape: {self.get_draw_attn.shape}")
            # 第一次prefill
            if self.draw_pic == 1 and raw_len_past_key_value!=self.config.num_hidden_layers:
                
                tensor_list = self.get_draw_attn.tolist()
                with open(output_datapath, "w") as json_file:
                    json.dump(tensor_list, json_file)
                if self.layer_idx == self.config.num_hidden_layers - 1:
                    logger.info(f"output_datapath: {output_datapath}")
                    assert 1==0
                pass
            # 第二次prefill
            elif self.draw_pic == 2 and raw_len_past_key_value==self.config.num_hidden_layers and q_len != 1:
                tensor_list = self.get_draw_attn.tolist()
                with open(output_datapath, "w") as json_file:
                    json.dump(tensor_list, json_file)
                if self.layer_idx == self.config.num_hidden_layers - 1:
                    logger.info(f"output_datapath: {output_datapath}")
                    assert 1==0
                pass
            # generate
            elif ( self.draw_pic == 3 or self.draw_pic == 5 )and raw_len_past_key_value==self.config.num_hidden_layers and q_len == 1:
                tensor_list = self.get_draw_attn.tolist()
                with open(output_datapath, "w") as json_file:
                    json.dump(tensor_list, json_file)
                if self.layer_idx == self.config.num_hidden_layers - 1:
                    logger.info(f"output_datapath: {output_datapath}")
                    assert 1==0
                # assert 1==0
                pass
            # 第二次prefill驱逐
            elif self.draw_pic == 4 and raw_len_past_key_value==self.config.num_hidden_layers and q_len != 1:
                tensor_list = self.get_draw_attn.tolist()
                with open(output_datapath, "w") as json_file:
                    json.dump(tensor_list, json_file)
                if self.layer_idx == self.config.num_hidden_layers - 1:
                    logger.info(f"output_datapath: {output_datapath}")
                    assert 1==0
                pass
            
        if past_key_value is not None: 
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if raw_len_past_key_value!=self.config.num_hidden_layers: # 第一次prefill
                if self.key_no_rope: # 保留不带位置编码信息的key
                    self.recent_attn=self.get_attn(raw_query_states, raw_key_states)
                    if self.kv_cache_eviction_prefill and not self.stage_eviction:
                        self._update_kv_first_prefill_key_no_rope(past_key_value,raw_key_states, before_rope_key_states,raw_query_states,raw_value_states, q_len)
                    else:
                        past_key_value.update(before_rope_key_states, raw_value_states, self.layer_idx, cache_kwargs)
                else:
                    self.recent_attn=self.get_attn(raw_query_states, raw_key_states)
                    if self.kv_cache_eviction_prefill and not self.stage_eviction:
                        # print(raw_len_past_key_value)
                        new_q_len = raw_key_states.shape[-2]
                        self._update_kv_first_prefill_batches(past_key_value,raw_key_states, raw_query_states,raw_value_states, new_q_len)
                    else:
                        past_key_value.update(raw_key_states, raw_value_states, self.layer_idx, cache_kwargs)
            else:  # 第二次prefill和generate
                if self.stage_eviction:
                    if kv_seq_len != 1: # 第二次prefill 且驱逐
                        # 更新self.window
                        self.window_size = min(self.window_size,kv_seq_len)
                        self._update_kv_second_prefill(past_key_value,raw_key_states, raw_query_states,raw_value_states, past_key_value.key_cache[0].shape[2])
                    else:
                        if self.kv_cache_dynamic:
                            attn_weights = torch.matmul(raw_query_states, raw_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                            if attention_mask is not None:  # no matter the length, we just slice it
                                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                                attn_weights = attn_weights + causal_mask
                            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                            self._update_kv_generate(attn_weights,past_key_value,raw_key_states,raw_value_states,self.layer_idx)
                elif self.kv_cache_eviction_prefill:
                    if self.kv_cache_dynamic:
                        attn_weights = torch.matmul(raw_query_states, raw_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                        if attention_mask is not None:  # no matter the length, we just slice it
                            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                            attn_weights = attn_weights + causal_mask
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                        
                        if kv_seq_len != 1: # 第二次prefill 多个token处理
                            self._update_kv_input(attn_weights,kv_seq_len)
                        else:
                            self._update_kv_generate(attn_weights,past_key_value,raw_key_states,raw_value_states,self.layer_idx)
        
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # print("attn_output.shape:{}".format(attn_output.shape))
        # print("attn_weights.shape:{}".format(attn_weights.shape))
        # print("past_key_value.shape:{}".format(past_key_value.shape))
        
        return attn_output, attn_weights, past_key_value

class LlamaForCausalLMPCW(LlamaForCausalLM, ABC):
    _no_split_modules = ["LlamaDecoderLayerPCW"]
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: LlamaConfig, 
                 capacity = 512, n_windows=1, kv_cache_eviction=False,
                 kv_cache_dynamic=False,stage_eviction=False,
                 get_recent_attn = False,key_no_rope=False,
                 draw_pic=0,attn_avg=False,in_eager_mode=False,
                 parallel_pattern=None,
                 calibration_mode=0,calibration_stage=None,
                 ):
        super(LlamaForCausalLM, self).__init__(config)
        # super().__init__(config)
        self.capacity =capacity
        self.n_windows = n_windows
        self.key_no_rope=key_no_rope
        self.parallel_pattern = parallel_pattern
        self.kv_cache_dynamic = kv_cache_dynamic
        # 确定config更改与否
        config.revise = False
        
        # raw code
        self.model = LlamaModelPCW(config, capacity = capacity, n_windows=n_windows, 
                                   kv_cache_eviction=kv_cache_eviction,
                                   kv_cache_dynamic=kv_cache_dynamic,
                                   stage_eviction=stage_eviction,
                                   draw_pic=draw_pic,
                                   calibration_mode=calibration_mode,
                                   calibration_stage=calibration_stage,
                                   in_eager_mode=in_eager_mode,
                                   attn_avg=attn_avg,
                                   get_recent_attn=get_recent_attn,
                                   key_no_rope=key_no_rope,
                                   )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        # 计算ppl
        if "ppl" in self.parallel_pattern and "default" not in self.parallel_pattern and labels is not None:
            # print("loss is:{}".format(loss))
            ppl = torch.exp(loss)
            print("ppl is:{}".format(ppl))
            assert 1==0
        else:
            label = None
        if labels is not None:
            # print("labels is not None")
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
      
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        windows_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        max_window_size: Optional[int] = 0,
        sum_windows_size: Optional[int] = 0,
        interval:Optional[float] = 1.0,
        interval_shift:Optional[int] = 0,
        position_dict:Optional[Dict] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):  
        # only last token for inputs_ids if past_key_values is defined in kwargs
        # new_length = input_ids.shape[-1]
        raw_input_ids = input_ids
        if past_key_values:
            input_ids = input_ids[:, -1:]
        attention_mask = kwargs.get("attention_mask")
        # 对于anchor类型，传入的是去掉了query部分长度的
        # 
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create PCW's position_ids on the fly 修改的
            position_ids = generate_pcw_position_ids(attention_mask, max_window_size, past_key_values,
                                                     sum_windows_size, windows_key_values,interval,interval_shift,
                                                     self.key_no_rope,position_dict)
            if position_ids.shape[-1] != 1:
                cache_position = torch.arange(sum_windows_size,sum_windows_size+raw_input_ids.shape[-1],device=position_ids.device)
            else:
                cache_position = torch.tensor([sum_windows_size+raw_input_ids.shape[-1]],device=position_ids.device)
        else:
            cache_position = kwargs.get("cache_position")
            if past_key_values:
                position_ids = torch.tensor(self.max_pos).unsqueeze(0).unsqueeze(0).to(cache_position.device)
                # print("position_ids:{}".format(position_ids))
                # print("cache_position:{}".format(cache_position))
                self.max_pos = self.max_pos+1
            else:
                self.max_pos = max_window_size
        
        # print(type(past_key_values))
        if self.kv_cache_dynamic and len(past_key_values) !=0 :
            attention_mask = attention_mask[:,-past_key_values[0][0].shape[2]-1:]
                    
        if windows_key_values and not past_key_values:
            print("ok")
            past_key_values = windows_key_values
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # print("past_key_values[0][0].shape:{}".format(past_key_values[0][0].shape))
            # print("len(past_key_values):{}".format(len(past_key_values)))

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "labels" : input_ids,
        }
        

class LlamaModelPCW(LlamaModel, ABC):
    
    def __init__(self, config: LlamaConfig, 
              capacity = 512, n_windows=1, kv_cache_eviction=False,
                 kv_cache_dynamic=False,stage_eviction=False,
                 get_recent_attn=False,key_no_rope=False,
                 draw_pic=0,attn_avg=False,in_eager_mode=False,
                 calibration_mode=0,calibration_stage=None,
                 ):
        super(LlamaModel, self).__init__(config)
        # super().__init__(config)
        # print("init LlamaModelPCW")
        # raw code
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerPCW(config, layer_idx, 
                                  capacity = capacity, n_windows=n_windows,
                                  kv_cache_eviction=kv_cache_eviction,
                                    kv_cache_dynamic=kv_cache_dynamic,
                                    stage_eviction=stage_eviction,
                                    draw_pic=draw_pic,
                                    calibration_mode=calibration_mode,
                                    calibration_stage=calibration_stage,
                                    in_eager_mode=in_eager_mode,
                                    attn_avg=attn_avg,
                                    get_recent_attn=get_recent_attn,
                                    key_no_rope=key_no_rope,
                                  ) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # # Initialize weights and apply final processing
        self.post_init()
        
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaFlashAttention2PCW,
    "flash_attention_2": LlamaFlashAttention2PCW,
    # "sdpa": LlamaSdpaAttentionPCW,
}

class LlamaDecoderLayerPCW(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None,
                 capacity = 512, n_windows=1, kv_cache_eviction=False,
                 kv_cache_dynamic=False,stage_eviction=False,
                 get_recent_attn=False,key_no_rope=False,
                 draw_pic=0,attn_avg=False,in_eager_mode=False,
                 calibration_mode=0,calibration_stage=None,
                 ):
        super(LlamaDecoderLayer, self).__init__()
        # print("init LlamaDecoderLayerPCW")
        # overriding attention:
        # print("config._attn_implementation:{}".format(config._attn_implementation))
        
        # raw code 
        
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx,
                                           capacity = capacity, n_windows=n_windows, 
                                           kv_cache_eviction=kv_cache_eviction,
                                           kv_cache_dynamic=kv_cache_dynamic,
                                           stage_eviction=stage_eviction,
                                           draw_pic=draw_pic,
                                           get_recent_attn=get_recent_attn,
                                           calibration_mode=calibration_mode,
                                           calibration_stage=calibration_stage,
                                           attn_avg=attn_avg,
                                           in_eager_mode=in_eager_mode,
                                           key_no_rope=key_no_rope,
                                           )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)