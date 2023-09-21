import torch
from typing import Optional, Tuple, List
from typing_extensions import TypedDict

from VisCPM.models.modules.embedding import EmbeddingExt
from VisCPM.models.modules.position_embedding import BucketPositionBias
from VisCPM.models.modules.transformer import Encoder
from VisCPM.models.modules.config import Config


class CPMBeeInferenceState(TypedDict):
    buffer_position: torch.Tensor
    buffer_context: torch.Tensor
    buffer_sample_ids: torch.Tensor
    buffer_num_segments: torch.Tensor
    buffer_segments: torch.Tensor
    buffer: List[Tuple[torch.Tensor, torch.Tensor]]


class CPMBeeConfig(Config):
    def __init__(
        self,
        vocab_size=30720,
        dim_model=4096,
        num_heads=64,
        dim_head=64,
        dim_ff=10240,
        num_layers=32,
        dropout_p=0.0,
        position_bias_num_buckets=256,
        position_bias_num_segment_buckets=256,
        position_bias_max_distance=2048,
        eps=1e-6,
        half: bool = True,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
    ):

        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_num_segment_buckets = position_bias_num_segment_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.eps = eps
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.vocab_size = vocab_size
        self.mask_modules = mask_modules


class CPMBeeTorch(torch.nn.Module):
    def __init__(self, config: CPMBeeConfig):

        super().__init__()
        self.config = config
        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            mask_modules=config.mask_modules,
        )

        self.input_embedding = EmbeddingExt(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            dtype=config.dtype,
            init_std=0.02,
        )

        # 一个可学习的position_bias,值由输入确定，不用管。光实现网络就行
        self.position_bias = BucketPositionBias(
            num_heads=config.num_heads,
            num_buckets=config.position_bias_num_buckets,
            num_segment_bucket=config.position_bias_num_segment_buckets,
            max_distance=config.position_bias_max_distance,
            dtype=config.dtype,
        )

    def forward(
        self,
        input: torch.Tensor,  # (batch, seqlen) int32
        input_sub: torch.Tensor,  # (batch, seqlen) int32
        length: torch.Tensor,  # (batch) int32
        context: torch.Tensor,  # (batch, seqlen) bool
        sample_ids: torch.Tensor,  # (batch, seq_len) int32
        num_segments: torch.Tensor,  # (batch, seq_len) int32
        segment: torch.Tensor,  # (batch, seqlen) int32
        segment_rel_offset: torch.Tensor,  # (batch, seq_len) int32
        segment_rel: torch.Tensor,  # (batch, num_segment_bucket) int32
        span: torch.Tensor,  # (batch, seqlen) int32
        ext_table_ids: torch.Tensor,  # (ext_table_size) int32
        ext_table_sub: torch.Tensor,  # (ext_table_size) int32
        hidden_states: torch.Tensor = None,
        **kwargs,
    ):
        batch = input.size(0)
        seqlen = input.size(1)
        # processing masks and position bias bucket
        with torch.no_grad():
            device = input.device

            # calc segment bucket
            segment_rel_2d = torch.masked_fill(
                segment[:, :, None] * num_segments[:, :, None]
                + segment[:, None, :]
                + segment_rel_offset[:, :, None],
                ~(
                    (sample_ids[:, :, None] == sample_ids[:, None, :])
                    & (span[:, None, :] == span[:, :, None])
                ),  # not in the same span or sample
                0,  # avoid torch.gather overflow
            ).view(batch, seqlen * seqlen)

            segment_bucket = torch.gather(
                input=segment_rel,
                dim=1,
                index=segment_rel_2d.long(),
            ).view(batch, seqlen, seqlen)

            segment_bucket.masked_fill_(
                ~(
                    (sample_ids[:, :, None] == sample_ids[:, None, :])
                    & (span[:, None, :] == span[:, :, None])
                ),  # not in the same span or sample
                1,  # bucket is used for in-context samples
            )

            # directional mask
            directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(
                seqlen, device=device
            ).view(-1, 1)
            # sample mask
            sample_mask_2d = (sample_ids[:, :, None] == 0) | (
                sample_ids[:, :, None] == sample_ids[:, None, :]
            )
            # context mask
            attention_mask = context[:, None, :] | (
                context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
            )
            # span mask
            attention_mask = (
                attention_mask & sample_mask_2d & (span[:, None, :] == span[:, :, None])
            )
            # length mask
            mask_1d = (
                torch.arange(seqlen, device=device)[None, :].repeat(batch, 1) < length[:, None]
            )
            attention_mask = (
                mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
            )
            position = torch.arange(seqlen, device=device).expand(batch, seqlen)

        if hidden_states is None:
            hidden_states = self.input_embedding(input, input_sub)
        position_bias = self.position_bias(position, position, segment_bucket)

        if kwargs.get('with_hidden_states', False):
            hidden_states, current_key_values, current_hidden_states = self.encoder(hidden_states, attention_mask,
                                                                                    position_bias, True)

            ext_table = self.input_embedding(ext_table_ids, ext_table_sub)

            logits = self.input_embedding.projection(hidden_states, ext_table)

            return logits, hidden_states, current_hidden_states
        else:
            hidden_states = self.encoder(hidden_states, attention_mask, position_bias, False)

            ext_table = self.input_embedding(ext_table_ids, ext_table_sub)

            logits = self.input_embedding.projection(hidden_states, ext_table)

            return logits, hidden_states

    def inference(
        self,
        input: torch.Tensor,  # (batch, len_q) int32        (beam_size,T)  首次输入，是q对应的所有segment的tokens
        input_sub: torch.Tensor,  # (batch, len_q) int32    (beam_size,T） 首次输入全0
        position: torch.Tensor,  # (batch, len_q)  int32    (beam_size,T） 首次输入，每条序列是query中各个token的pos。[0,1,2,...107]
        context: torch.Tensor,  # (batch, len_q) bool       (beam_size,T） 全1          
        sample_ids: torch.Tensor,  # (batch, len_q) int32   (beam_size,T） 首次输入，全0
        num_segments: torch.Tensor,  # (batch, len_q) int32 (beam_size,T）全9。 共9个segment
        segment: torch.Tensor,  # (batch, len_q) int32       (beam_size,T）是input token对应的segment_id。[0,0,1,1,1,2,2,2,...7,7,7]
        segment_rel_offset: torch.Tensor,  # (batch, len_q) int32         #  (beam_size,T）全0
        segment_rel: torch.Tensor,  # (batch, num_segment_bucket) int32   #  (beam_size,n_seg^2） 每个seg和其他seg之间的关系。9个seg,81个关系。[3,81]
        ext_table_ids: torch.Tensor,  # (ext_table_size) int32   []
        ext_table_sub: torch.Tensor,  # (ext_table_size) int32   []
        
        past_key_values: Optional[CPMBeeInferenceState] = None,  # 构建阶段。没有KV cache,传入None
        
        hidden_states: torch.Tensor = None,     # (beam_size,T,d)   3个输入序列的embed,已经经过rotary_embed(nlp)、bei得到q序列的隐向量
    
    ) -> Tuple[torch.Tensor, torch.Tensor, CPMBeeInferenceState]:
        with torch.no_grad():
            if past_key_values is None:
                # 首次，没有KVcache.都是原始query的信息
                present_position = position
                present_context = context
                present_sample_ids = sample_ids
                present_num_segments = num_segments
                present_segments = segment
                present_buffer = None
            else:
                # 下一次，input，position，context等，都是序列当前最后一个token的状态
                #  past_key_values则是上次的输入序列的状态
                #  拼起来以后，得到的present*,是完整的输入序列对应的状态，包含q和已经预测出的tokens。是当前的k,v.
                present_position = torch.cat([past_key_values["buffer_position"], position], dim=-1) # 新的k,v，含已经预测出的tokens
                present_context = torch.cat([past_key_values["buffer_context"], context], dim=-1)
                present_sample_ids = torch.cat(
                    [past_key_values["buffer_sample_ids"], sample_ids], dim=-1
                )
                present_num_segments = torch.cat(
                    [past_key_values["buffer_num_segments"], num_segments], dim=-1
                )
                present_segments = torch.cat([past_key_values["buffer_segments"], segment], dim=-1)
                # 上次每层计算att时，用过的完整K矩阵,V矩阵。(B,T,d），还不包含本次这个新token
                present_buffer = past_key_values["buffer"]

            batch = input.size(0)   # B
            len_q = input.size(1)   # Tq 首次是原始输入序列长度。 之后每个新token作为q,得到该token对所有kv的hidden_state
            len_buffer = present_position.size(1) # T 当前预测到哪一步了.对于首个step,是query本身的长度。
                                                  #                   后续包含了新预测出来的token,是kv完整长度
            # (beam_size,Tq,T_cur)
            # query每个token所在的segment,和当前序列kv,每个token的segment之间的关系
            segment_rel_2d = torch.masked_fill(
                segment[:, :, None] * num_segments[:, :, None]    # query中每个token的segment,乘上总的segment数量(9) (beam_size,Tq,1) [9,9,9,18,18,...63,63]
                + present_segments[:, None, :]                    # (beam_size,1,Tv)
                + segment_rel_offset[:, :, None],
                ~(
                    (sample_ids[:, :, None] == present_sample_ids[:, None, :])
                ),  # not in the same sample
                0,  # avoid torch.gather overflow
            ).view(batch, len_q * len_buffer)

            # query中每个token,与当前序列每个token的segment的关系。（beam_size,Tq,T_cur）
            segment_bucket = torch.gather(
                input=segment_rel,
                dim=1,
                index=segment_rel_2d.long(),
            ).view(batch, len_q, len_buffer)

            segment_bucket.masked_fill_(
                ~(
                    (sample_ids[:, :, None] == present_sample_ids[:, None, :])
                ),  # not in the same span or sample
                1,  # bucket is used for in-context samples
            )

            # directional mask
            # （Beam,1,T_cur）   (Beam,Tq,1)
            #                          0,1,...i,i+1,  Tcur
            # q中每个token：  token_i   1 1    1 0 0 0  0      只能计算当前序列i之前的att
            directional_mask_2d = present_position[:, None, :] <= position[:, :, None]  # （beam_size,Tq,T_cur）
            # sample mask
            sample_mask_2d = (sample_ids[:, :, None] == 0) | (   # 目前都是True
                sample_ids[:, :, None] == present_sample_ids[:, None, :]
            )
            # context mask
            attention_mask = present_context[:, None, :] |(         #首次全是1. 可以对所有q att。
                context[:, :, None].logical_not()
                & directional_mask_2d.view(batch, len_q, len_buffer) # att_score(Tq,Tl),Tl中可以被mask的地方，置为true
            )
            # span mask
            attention_mask = attention_mask & sample_mask_2d
            # length mask
            mask_1d = present_num_segments != 0
            attention_mask = mask_1d.view(batch, 1, len_buffer) & attention_mask  # q只算该新token [B,1,tk]
            if hidden_states is None:
                # 后续每个step,只传入一个token.得到该token的embed
                hidden_states = self.input_embedding(input, input_sub)  # (B,1,d)

            # 按每个q和v的segment间的关系，按数量级(bucket)得到一个embed,heads维
            # 中间计算经过一个自定义的网络。
            # 最终得到的是(B，heads,Tq,Tk) 的embed,后续可以加在att_score上。用来补充segment带来的qk间关系.
            # 每个step,是(B,heads,1,Tk)。 q只算该新token
            position_bias = self.position_bias(position, present_position, segment_bucket) # 经过一个网络，计算得到。(B，heads,Tq,Tk)
            
            # 首个step：
            #   hidden_states:最终本step得到的hidden_state (B,T,d)
            #   current_key_values[layer]:每层计算att时，用到的映射过的完整k,V矩阵。首次是（B,Tq,d）.保存下来，后续用
            # 后续每个step,用当前序列的最后一个位置做q:
            #   hidden_states:最后一个token得到的hidden_state (B,1,d),用来预测next token
            #   present_key_values:截止当前的完整序列（含query和预测出的n个token），本次使用的KV矩阵。下次拼上新token对应的kv向量后，可以直接用
            hidden_states, present_key_values, _ = self.encoder(
                hidden_states,   # (B,Tq,d) 当前3条完整输入序列，对应的hidden_states。 后续每个step,只用一个新token做q（B,1,d）
                attention_mask,  # (B,Tq,Tv)  用来对att_score做mask. (softmax前). 后续每个step,只用一个新token做q（B,1,Tv）
                position_bias,   # (B,heads,Tq,Tv)  用来补充att_score.  后续每个step,只用一个新token做q.(B,heads,1,Tk)
                True,            # usecache==True
                present_buffer,  # 首次是None.之后是上次计算att时，每层用过的完整K,V矩阵。(B,T,d），还不包含本次作为q的这个新token
            )
            # ext_table_ids：[]  int32   用来lookup
            # ext_table_sub：[]  int32   用来rotate
            ext_table = self.input_embedding(ext_table_ids, ext_table_sub) # forward. 这里是[]。本应得到（B,T,d）,lookup到一些embed
            
            # 本次step
            #（B,T,V）  和词典做交互后，当前每个输入token,得到的logits(1,V). 首次是每个q,得到一个(1,V)
            # 后续每个step,hidden_states是当前最后一个位置得到的(B,1,d)。和词表映射后(d,V)，得到logits(B,1,V),用来预测下一个token
            logits = self.input_embedding.projection(hidden_states, ext_table) # （B,T,V）

            return (
                logits,               # 首次是（B,Tq,V）。是q中每个token预测的logits
                                      # 后续是和词表映射后(d,V)，得到logits(B,1,V),用来预测下一个token
                hidden_states,        # 最终本step得到的hidden_state.首次是 (B,Tq,d),是每个token的hidden_state
                
                # past_key_values
                {
                    # 首次是原始序列q对应的信息  （beam_size,Tq）
                    "buffer_position": present_position, 
                    "buffer_context": present_context,
                    "buffer_sample_ids": present_sample_ids,
                    "buffer_num_segments": present_num_segments,
                    "buffer_segments": present_segments,
                    # 每层计算att时，用到的映射过的完整K矩阵,V矩阵。(B,T,d）
                    # 首次，是每层映射过的K(B,Tq,d)，V(B,Tq,d).保存下来，后续用. 在本次计算时，直接用Q和该K,V相乘。
                    "buffer": present_key_values, 
                                                  
                },
            )
