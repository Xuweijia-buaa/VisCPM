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
import math
import torch.nn.functional as F
from .position_embedding import RotaryEmbedding
from typing import Optional


class Embedding(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):

        super().__init__()

        self.dim_model = embedding_size
        self.weight = torch.nn.parameter.Parameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype)  # 作为embedding的权重。初始化时，作为模型参数，load进来。
        )

    def forward(self, ids: torch.Tensor):
        """
        Args:
            ids (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        # 前向，直接用torch的embedding函数。
        # 用固定weights填充得到embed,用ids来lookup.  得到的embed(B，T,d), 用维度d归一化
        embeds = F.embedding(ids, self.weight) / math.sqrt(self.dim_model)  # 返回ids(B,T), lookup得到的（B,T,d）  （归一化后）
        return embeds

    def projection(self, x: torch.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501
        
        # 输入x(B,T,d),沿维度d归一化后。经过linear层
        # 相当于计算：y = xA^T + b。 这里没有设置b。 A由embed矩阵提供
        # 相当于x(B,T,d) * embed(d,V)  ->  logits(B,T,V),返回
        logits = F.linear(x / math.sqrt(self.dim_model), self.weight)
        return logits


class EmbeddingExt(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,#4096
        dtype: torch.dtype = torch.half,  # 默认是torch.float16。 conflig里也是
        init_mean: float = 0.0,
        init_std: float = 1,
        distance_scale: int = 16,
    ):

        super().__init__()

        self.dim_model = embedding_size
        self.rotary_emb = RotaryEmbedding(
            dim=embedding_size, distance_scale=distance_scale, dtype=dtype
        )
        # distance_scale:16 默认。剩下的都在config里，也固定

        self.weight = torch.nn.parameter.Parameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),  # embed的矩阵
        )

    def forward(self, ids: torch.Tensor, ids_sub: torch.Tensor):
        """
        Args:
            ids (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            ids (:obj:`torch.Tensor` of shape ``(batch_size)``): Subscript of input sequence tokens.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        # 类似上边，首先通过ids(B,T)和embed_weight,lookup得到embedding(B,T,d)
        embeds = F.embedding(ids, self.weight) / math.sqrt(self.dim_model)  #  (B,T,d)  -> ([1, 108, 4096])
        # 得到语言tokens,原始embed，对应的rotate embeding（B,T,d）
        return self.rotary_emb(embeds, ids_sub)   # 其中参数都是固定值。没有大embedding矩阵了。

    def projection(self, x: torch.Tensor, ext_table: Optional[torch.Tensor] = None):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
            ext_table (:obj:`torch.Tensor` of shape ``(ext_table_size, dim_model)``): Ext vocab table.
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_size + ext_table_size)``: The projection output.
        """  # noqa: E501
        # 类似上边，x(B,T,d)，首先经过embed构成的linear层，得到logits
        # 相当于x(B,T,d) * embed(d,V)  ->  logits(B,T,V),
        logits = F.linear(x / math.sqrt(self.dim_model), self.weight) # （B,T,V）  和词典做交互后，得到的logits
        #
        if ext_table is not None:
            logits_ext = F.linear(x, ext_table)  # input,weights. 相乘
            logits = torch.cat([logits, logits_ext], dim=-1)
        return logits
