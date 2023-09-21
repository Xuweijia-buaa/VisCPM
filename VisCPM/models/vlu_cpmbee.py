from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from timm.models.layers import trunc_normal_
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import ModelOutput

from VisCPM.models.cpmbee import CPMBeeTorch
import os


def construct_query_parameter(query_k, h_size, init_weights):
    query_data = torch.zeros(query_k, h_size)
    trunc_normal_(query_data, std=.02)
    for idx in range(query_k):
        if init_weights[idx] is not None:
            query_data[idx] = init_weights[idx]
    query = torch.nn.Parameter(query_data)
    return query


@dataclass
class CausalVLLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class VLU_CPMBee(torch.nn.Module):
    def __init__(self, llm: CPMBeeTorch, vpm, vision_dim, query_num, device=None) -> None:
        super().__init__()
        self.device = device
        self.vpm = vpm
        self.llm = llm

        self.vision_dim = vision_dim
        self.query_num = query_num  #64
        self.query = None

        if query_num is not None: # 走这个分支
            bos_weight = self.vpm.beit3.text_embed.weight.data[0]
            eos_weight = self.vpm.beit3.text_embed.weight.data[2]
            query_init_weight = [bos_weight] + [None] * (self.query_num - 2) + [eos_weight]
            self.query = construct_query_parameter(   # 初始化网络时就构建了.torch.Size([64, 1024])  float32
                self.query_num, self.vision_dim, query_init_weight) 


        # self.vlu_cpmbee.mapping
        # Sequential(
        # (0): Linear(in_features=1024, out_features=4096, bias=True)
        # (1): GELU(approximate='none')
        # (2): Linear(in_features=4096, out_features=4096, bias=True)
        # )
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(self.vpm.hidden_size, self.llm.config.dim_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.llm.config.dim_model, self.llm.config.dim_model)
        )

    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            pixel_values = data['pixel_values']
            # TODO:
            # 这里使用beit3模型，forward. 传入图片，query_embed,传出图像编码
            # pixel_values：torch.Size([1, 3, 224, 224])  原始图像
            # query_embed： torch.Size([64, 1024])
            # 得到视觉编码状态：torch.Size([1, 64, 1024])  （Q,d=1024）
            vision_hidden_states = self.vpm(pixel_values=pixel_values, query_embed=self.query) # 得到
            # 经过线性层，映射到语言模型维度 （Q,h=4096）
            # vision_hidden_states: torch.Size([1, 64, 4096])
            vision_hidden_states = self.mapping(vision_hidden_states)  # (query_num, llm_dim)
        else:
            vision_hidden_states = data['vision_hidden_states']

        # 输入的语言，经过rotate_embed。得到语言tokens,原始embed，对应的rotate embeding（B,T,d）
        vllm_embedding = self.llm.input_embedding(data['input_ids'], data['input_id_subs']) # (1,108,d=4096)
        # 输入图片经过beit3模型，得到的embed （B,64,d）
        vision_hidden_states = vision_hidden_states.type(vllm_embedding.dtype)              # (1,64,d=4096)

        # tokens里边，image字段对应的segment的start,end [7,71],64个
        # 把这64个tokens对应的index,实例化出来。[7,8,...70],对应tokens中，image tokens的index   int64
        image_bound = data['image_bound']    
        image_bound = image_bound.squeeze(1)  # shape[1,2]    tokens里边，image字段对应的segment的start,end [7,71],64个
        image_indices = torch.stack(          #[1,64] 
            [torch.arange(r[0], r[1], dtype=torch.long) for r in image_bound] # shape是[1.64]  内容是image tokens的index:[7,8,...70]   int64
        ).to(self.device)

        # 将得到的视觉embed, 填充到给image留着的tokens中. 一共64个tokens
        # 视觉embed,被填充到vllm_embed的位置不定（取决于segment被解析的顺序。是data dict中k == "image"的孩子节点。）
        # 但数目固定，是query_num==64个。用beit3模型得到的64个embed填充。
        # 可以通过对入参的解析，把这些位置放最前/最后，从而填充到固定位置。 （即使得到这里的image_indices是[0-63]/[T-63,T]）
        # 填充位置(1,64)，最后一维拓展为 -> (1,64,d)
        # 原始embed(B,T,d) ,按该index(B,64,d)，找对应位置，填充image embed
        vllm_embedding.scatter_(1, image_indices.unsqueeze(-1).repeat(1, 1, vllm_embedding.shape[-1]),
                                vision_hidden_states)

        return vllm_embedding, vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        logits, hidden_states = self.llm(
            input=data['input_ids'],
            input_sub=data['input_id_subs'],
            length=data['length'],
            context=data['context'],
            sample_ids=data['sample_ids'],
            num_segments=data['num_segments'],
            segment=data['segment_ids'],
            segment_rel_offset=data['segment_rel_offset'],
            segment_rel=data['segment_rel'],
            span=data['span'],
            ext_table_ids=data['ext_table_ids'],
            ext_table_sub=data['ext_table_sub'],
            hidden_states=vllm_embedding
        )

        return CausalVLLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            vision_hidden_states=vision_hidden_states
        )
