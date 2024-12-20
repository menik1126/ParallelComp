import math
from abc import ABC
from typing import Optional, Tuple, Dict
import torch.nn.functional as F
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, LlamaRMSNorm, \
    LlamaDecoderLayer, LlamaModel, LlamaForCausalLM

from pcw_wrapper import generate_pcw_position_ids

"""
The following code is mainly copy+paste from the original modelling_llama.py:
LlamaAttention uses a caching mechanism for the positional rotation vectors (using LlamaRotaryEmbedding). 
This mechanism forces us to override LLaMa attention layer, which in turn forces us to override the decoder, 
and model (so that the correct forward function would be called).
"""


class LlamaForCausalLMPCW(LlamaForCausalLM, ABC):
    _no_split_modules = ["LlamaDecoderLayerPCW"]

    def __init__(self, config: LlamaConfig, capacity = 512, n_windows=1):
        super(LlamaForCausalLM, self).__init__(config)
        # print("capacity:{}".format(capacity))
        # assert 1==0
        self.capacity =capacity
        self.n_windows = n_windows
        # print("self.n_windows:{}".format(self.n_windows))
        # assert 1==0
        # using our Llama model variant:
        self.model = LlamaModelPCW(config, capacity = capacity, n_windows=n_windows)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.LongTensor,
                                      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      windows_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      max_window_size: Optional[int] = 0,
                                      sum_windows_size: Optional[int] = 0,
                                      **kwargs
                                      ) -> Dict:
        """input_ids:
            ids of task_tokens.
         attention_mask:
            concatenation of windows + task tokens attentions masks.

         Note (past_key_values vs windows_key_values):
             In the first token generation, past_key_values is None while windows_key_values contains the combined past
             key values of context windows. During following generations, past_key_values is the concatenation of
             windows_key_values + previous generations. Thus, windows_key_values is practically ignored.
             """

        # only last token for inputs_ids if past_key_values is defined in kwargs
        new_length = input_ids.shape[-1]
        if past_key_values:
            input_ids = input_ids[:, -1:]
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            
            # create PCW's position_ids on the fly 修改的
            position_ids = generate_pcw_position_ids(attention_mask, max_window_size, past_key_values,
                                                     sum_windows_size, windows_key_values)
            
            # # create position_ids on the fly for batch generation 原始的
            # position_ids = attention_mask.long().cumsum(-1) - 1
            # position_ids.masked_fill_(attention_mask == 0, 1)
            # if past_key_values:
            #     position_ids = position_ids[:, -1].unsqueeze(-1)

        if windows_key_values and not past_key_values:
            past_key_values = windows_key_values

        # print("input_ids.shape:{}".format(input_ids.shape))
        # print("position_ids.shape:{}".format(position_ids.shape))
        # print("attention_mask.shape:{}".format(attention_mask.shape))
        #n_windows = 2
        attention_mask = attention_mask[:, :self.capacity+(self.capacity-1)*(self.n_windows-1)+new_length]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


class LlamaModelPCW(LlamaModel, ABC):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, capacity = 512, n_windows=1):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # using the alternative decoder layer:
        self.layers = nn.ModuleList([LlamaDecoderLayerPCW(config, capacity = capacity, n_windows=n_windows) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class LlamaDecoderLayerPCW(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, capacity = 512, n_windows=1):
        super().__init__(config)
        # overriding attention:
        self.self_attn = LlamaAttentionPCW(config=config, capacity = capacity, n_windows=n_windows)


class LlamaAttentionPCW(LlamaAttention):
    def __init__(self, config: LlamaConfig, capacity = 512, n_windows=1):
        super().__init__(config)
        # overriding attention:
        self.capacity = capacity
        # print("self.capacity:{}".format(self.capacity))
        # assert 1==0
        self.n_windows = n_windows

    # we have to override the forward attention due to the rotary embeddings caching mechanism
    # 在这里修改旋转位置编码
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # *** changes to the original code to accommodate PCW:
        # making sure that the model generates rotary embeddings in the correct length:
        # print("position_ids:{}".format(position_ids))
        # print("position_ids:{}".format(position_ids.shape))
        # print("torch.max(position_ids):{}".format(torch.max(position_ids)))
        seq_len = kv_seq_len if position_ids is None else int(torch.max(position_ids) + 1)
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # *** End of changes due to PCW, the rest of the function is copy-paste from the original transformer package.


        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None :
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
           raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        
        # print("key_states.shape:{}".format(key_states.shape))
        
        if past_key_value is not None : #generate阶段
            past_key_value = (key_states, value_states) if use_cache else None
        else: #prefill阶段
            # 压缩kv
            head_dim = self.head_dim
            window_size = 4
            kernel_size = 7
            pooling = 'avgpool'
            attn_weights_sum = attn_weights[:, :, -window_size:, :-window_size].sum(dim = -2)
            if pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
            elif pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            #print("self.capacity0:{}".format(self.capacity))
            #assert 1==0
            top_k = self.capacity - window_size
            indices = attn_cache.topk(top_k, dim=-1).indices
            indices = indices.sort(dim=-1).values
            indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  
            k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices_expanded)
            v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices_expanded)
            k_cur = key_states[:, :, -window_size:, :]
            v_cur = value_states[:, :, -window_size:, :]
            revise_key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            revise_value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            past_key_value = (revise_key_states, revise_value_states) if use_cache else None
            
        attn_output = torch.matmul(attn_weights, value_states).to(query_states.dtype)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
