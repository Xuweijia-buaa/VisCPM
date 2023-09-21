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

from typing import Optional, Tuple
import torch
import math
from .linear import Linear


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        dtype: torch.dtype = torch.half,
        dropout_p: Optional[float] = None,
    ) -> None:

        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.project_q = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)     # QkV对应的映射矩阵。 X->QKV
        self.project_k = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_v = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)

        self.attention_out = Linear(self.num_heads * self.dim_head, self.dim_model, dtype=dtype) # att中的out，融合multi_heads

        self.softmax = torch.nn.Softmax(dim=-1)

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_q: torch.Tensor,   # （B,T,d）
        hidden_kv: torch.Tensor,  #  (B,T,d)   和q的hidden_state相同
        attention_mask: torch.BoolTensor, # (B,Tq,Tv)  用来对att_score做mask. (softmax前).首次Tq==Tv
        position_bias: torch.Tensor,      # (B,heads,Tq,Tv)  用来补充att_score
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # 首次是None
    ):
        """
        Args:
            hidden_q (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501

        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(1)
        len_k = hidden_kv.size(1)

        # 首次，qkv对应的hidden相同。都是输入序列embed得到的。(B,Tq,d)
        #     分别乘对应映射矩阵。得到Q,K,V  (B,Tq,nhead*d)
        # 之后每个step:
        #     q是当前最后一个输出对应的token. 而hidden_q,hidden_kv相同,都是该token的原始embed
        #     只需要分别映射原始embed,为q,k,v.  q不变，k,v拼到原始序列的大K,V矩阵上。
        h_q = self.project_q(hidden_q)
        h_k = self.project_k(hidden_kv)
        h_v = self.project_v(hidden_kv)

        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)  # 转成(B,nhead,T,d)  后续是（B,nhead,1,d）
        h_k = h_k.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        h_v = h_v.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        if past_kv is not None:
            # 上次的完整序列算出的KV（B,nhead,Tv-1,d）,拼上这个新token的kv（B,nhead,1,d）:(B,nhead,Tv,d) 是当前完整序列映射好的KV
            # 新token q, 对完整序列做att
            h_k = torch.cat([past_kv[0], h_k], dim=-2)    
            h_v = torch.cat([past_kv[1], h_v], dim=-2)
            len_k = h_k.size(-2)
            
        # 除了首次，后续每个step,Tq=1。算出的hidden是（B,1,d）,用来预测next token

        # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
        # 计算注意力矩阵（Tq,Tv）
        # Q*K.T /sqrt(d)  -> att_score(B,nhead,Tq,Tv)
        score = torch.matmul(h_q, h_k.transpose(-1, -2)) / math.sqrt(self.dim_head)
        
        # 我们算出来的一个参数（B,head,Tq,Tv），加到注意力矩阵上
        score = score + position_bias    # attention score矩阵, 加上这个位置bias。 得到新的attention_score (Tq,Tk)
        
        # 对att_score做mask
        # 经过mask,把mask中False的位置，置-inf,再做softmax，值就是0了。tq不关注之后的mask。
        score = torch.masked_fill(       
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == False,  #原始mask是（B,Tq,Tv）
            torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
        )

        score = self.softmax(score)   # （B,Tq,Tv）

        # softmax后，原来mask掉的位置，score填充成0 （防止数值不稳定？）
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype),
        )

        if self.dropout is not None:
            score = self.dropout(score)

        # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
        # QK（Tq,Tk）*V(Tk,d)   聚合，得到每个hidden新的表示  （B,nhead,Tq,d）
        score = torch.matmul(score, h_v)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3) #转成（B,Tq,nhead,d）
        score = score.contiguous().view(batch_size, len_q, self.num_heads * self.dim_head)       # （B,Tq,nhead*d）

        score = self.attention_out(score)  # 本次输出（B,Tq,nhead*d） -> (B,Tq,d). 
        if use_cache:
            return score, (h_k, h_v)  # 输出本次的hidden_state (B,Tq,d_model), 
                                      # 以及本层layer计算过程中的K,V矩阵。是原始hidden映射后的：（B,Tv,d）。 首次Tv==Tq
                                      
                                      # 后续的hk,hv,是截止当前的完整序列（含query和预测出的n个token），使用的KV矩阵。下次直接用（拼上新的kv向量后），不重新映射。
                                      
        # 除了首次，后续每个step,Tq=1。算出的score是（B,1,d）,用来预测next token
        else:
            return score
