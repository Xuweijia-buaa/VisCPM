# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Optional, List, Tuple

from .blocks import TransformerBlock
from .layernorm import LayerNorm


class Encoder(torch.nn.Module):
    """Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-6.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        dtype: torch.dtype = torch.half,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
    ):

        super().__init__()

        self.num_layers = num_layers

        if mask_modules is not None:
            assert (
                len(mask_modules) == num_layers
            ), "The total number of masks should equal to num_layers"
            for mask_module in mask_modules:
                assert (
                    len(mask_module) == 2
                ), "For encoder, each mask should be (mask_att, mask_ffn)"
        else:
            mask_modules = [(False, False)] * num_layers

        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    dim_model=dim_model,
                    dim_ff=dim_ff,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    dtype=dtype,
                    eps=eps,
                    dropout_p=dropout_p,
                    mask_att=mask_modules[ith][0],  # 每层2个mask.分别用来mask attention和ffn. 这里配置是否掉其中的一个block.默认都要
                    mask_ffn=mask_modules[ith][1],
                )
                for ith in range(num_layers)
            ]
        )

        self.output_layernorm = LayerNorm(dim_norm=dim_model, dtype=dtype, eps=eps) # rms_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,    # (B,Tq,d) 当前3条完整输入序列，对应的hidden_states
        attention_mask: torch.Tensor,   # (B,Tq,Tv)  用来对att_score做mask. (softmax前).首次Tq==Tv
        position_bias: torch.Tensor,    # (B,heads,Tq,Tv)  用来补充att_score
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, # 首次是None
    ):
        """
        Args:
            hidden-states (:obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``): Input of encoder, might be the embedding of a batch of sequences.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_enc, seq_enc)``): Avoid invalid areas to participate in the calculation
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_enc, seq_enc)``) Provides position information to attention mechanism.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``: The encoder output.

        """  # noqa: E501
        if not use_cache:
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask, position_bias)
            hidden_states = self.output_layernorm(hidden_states)
            return hidden_states
        else:
            with torch.no_grad():
                current_key_values = []
                current_hidden_states = []
                # 48个TransformerBlock。 每个包含一个sef-att(SelfAttentionBlock)和一个ffn (FFNBlock)
                # 每个self-att:包含一个LayerNorm，一个Attention,一个dropout
                # 每个ffn:包含linear,gatedGelu等
                for i, module in enumerate(self.layers): 
                    hidden_states = module(
                        # 除了首次encode（B,Tq,d）.后续每个step，只用一个新token做q 
                        hidden_states,      # （B,Tq,d）。后续只用新token作为q.传入该token的原始embed:（B,1,d）
                        attention_mask,     #  (B,Tq,Tv) 对att_score做mask.(softmax前). 后续每个step,只用一个新token做q（B,1,Tv）
                        position_bias,      # (B,heads,Tq,Tv)  用来补充att_score.  后续每个step,只用一个新token做q.(B,heads,1,Tk)
                        past_key_value=past_key_values[i] if past_key_values else None, # 首次是None。
                                                                                        # 之后是保存的，该层上次使用的大K,V矩阵
                                                                                        # 不包含本次作为q的这个新token。该token还需要重新映射q，k,v
                        use_cache=use_cache,
                    )
                    if use_cache:
                        # 首次：
                        #  current_hidden_states[layer]: (B,Tq,d)  每层transofrmer结果。 长度同输入序列hidden_states的T
                        #                                          是该层原始的输入序列q,经过映射，分别得到(Q,K,V),att的结果                                 
                        # current_key_values[layer]:是本层计算att时，用到的映射过的k,V矩阵。（B,Tq,d）.保存下来，后续用
                        current_key_values.append(hidden_states[1])
                        current_hidden_states.append(hidden_states[0])
                        hidden_states = hidden_states[0]
                # 最后边一个RMS_LayerNorm Block
                hidden_states = self.output_layernorm(hidden_states) # 作为下一层的输入，输出到下一层Transformer中
                if use_cache:
                    # hidden_states:最终本step得到的hidden_state (B,T,d)
                    # current_hidden_states[layer]:每层得到的hidden_states (B,T,d)
                    # current_key_values[layer]:本层计算att时，用到的映射过的k,V矩阵。首次是（B,Tq,d）.保存下来，后续用
                    return hidden_states, current_key_values, current_hidden_states
                else:
                    return hidden_states
