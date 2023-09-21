from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import os

from VisCPM.generation.generation_utils import BeamHypotheses, apply_repetition_penalty
from VisCPM.cpm_tokenizers.bee import CPMBeeTokenizer
from VisCPM.models import VLU_CPMBee
from VisCPM.models.cpmbee import CPMBeeTorch
from VisCPM.utils.utils import convert_data_to_id, pad

# 一个beam_search的套子
class VLLMCPMBeeGeneration:
    def __init__(self, model: VLU_CPMBee, tokenizer: CPMBeeTokenizer, transform, device):
        model.eval()
        self.model = model       # VLU_CPMBee对象。是一个torch.nn.Module。 包含llm模型和vpm模型
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device

    def _convert_to_tensors(self, data: Any, in_context_samples: List[Any] = [], max_inp_length: Optional[int] = None):
        answer_placeholders = []

        def _put_placeholder(data: Any, path: List[str] = []):
            if isinstance(data, dict):
                ret = {}
                for k, v in data.items():
                    ret[k] = _put_placeholder(v, path + [k])
                return ret
            else:
                answer_placeholders.append(path)
                return "<ans_{}>".format(len(answer_placeholders))

        data["<ans>"] = _put_placeholder(data["<ans>"])
        (
            input_ids,          # 每个seg对应的tokens，编码后，都拼到一起。 含img对应的<unk>，ques标签，编码后的内容
            input_id_subs,
            context,
            segment_ids,
            segment_rel,
            n_segments,
            table_states,     # {'ext_table': {86583: '<ans_1>'}, 'token_id_table': {'<ans>': {86583: 0}} }
            image_bound,
        ) = convert_data_to_id(self.tokenizer, data, shuffle_answer=False, max_depth=8)
        # 每个seg,对应的token,前后加上开始结束标志
        # ids: [...                                                                            ]   编码好的tokens. id_subs全0
        # segs:[0,0,0, 1,1,1  2,2,2,2,2,2,2,2  3 3 3    4 4  5 5 5       6 6....    7,7,7,  8,8]
        # 对应 [root,  image, <unk><unk><unk>, 'context',"", "question", "如果用...", <ans>, <ans_1>]
        #                    64个token.这段留着，后续用img_embed填充                            这个需要predict
        #                    image_bound：[7,71],去掉头尾的真实img填充的64个tokens
        # context:[1 1 1   ...                                                      1 1 1,  0,0 ]  需要预测的这个seg,对应位置置为0
        # segment_rel
        # 对应segment_bound[i],是该segment在tokens中的位置
        # segment_bound[0]:(0,3)      segment_bound[1]:(3,6)   segment_bound[2]:(6,72) (对应图像tokens)        
        # num_segments：9 
        # segment_rel: 81个，是每个seg和其他seg之间根据树中深度，得到的关系
        # array([ 0,  2,  3,  2,  3,  2,  3,  2,  3, 
        #         9,  0,  2, 10, 11, 10, 11, 10, 11,     seg[1]，和其他seg在树中的关系。越小越近。和自己是0
        #        17,  9,  0, 18, 19, 18, 19, 18, 19, 
        #        9, 10, 11,  0,  2, 10, 11,  10, 11, 
        #        17, 18, 19,  9,  0, 18, 19, 18, 19,
        #        9, 10, 11, 10, 11,  0,  2, 10, 11, 
        #        17, 18, 19, 18, 19,  9,  0, 18, 19, 
        #        9, 10, 11, 10, 11,  10, 11,  0,  2, 
        #        17, 18, 19, 18, 19, 18, 19,  9,  0], dtype=int32)         

        if max_inp_length is not None:
            input_ids = input_ids[: max_inp_length]
            context = context[: max_inp_length]
            segment_ids = segment_ids[: max_inp_length]

        sub_ans_map: Dict[int, int] = {}
        for fake_id, token_sub in table_states["token_id_table"]["<ans>"].items(): # {'<ans>': {86583: 0}} }
            # fake_id, token_sub： {86583: 0}
            # token_sub是0，即<ans_1>在拓展词表中的id
            token = table_states["ext_table"][fake_id]   # 'ext_table': {86583: '<ans_1>'}。答案token '<ans_1>'
            if token.startswith("<ans_") and token.endswith(">"):
                ans_id = int(token[5:-1])         # 1
                sub_ans_map[token_sub] = ans_id   # <ans_1>   第1个答案。ans_id是1     <ans_2>, ans_id是2

        tmp_input_ids = []
        tmp_input_sub = []
        tmp_input_seg = []

        predict_segments: List[Tuple[int, int]] = []
        for i in range(input_ids.shape[0]):
            if context[i] == 0:  # 待预测的输出segment <ans_1>。 segment8,单独拆出来
                if input_ids[i] == self.tokenizer.encoder["<ans>"]: # 9
                    # is ans
                    # (segment_id, ans_id)
                    # token i对应的是答案。所在的segment，作为被预测的segment。同时加上该额外token的第几个答案id. 这里是1。第一个答案
                    predict_segments.append((segment_ids[i], sub_ans_map[input_id_subs[i]]))
            else:
                # 把待预测的segment拆了出来。剩下的segment对应的信息
                tmp_input_ids.append(input_ids[i])
                tmp_input_sub.append(input_id_subs[i])
                tmp_input_seg.append(segment_ids[i])

        if len(predict_segments) == 0:
            raise ValueError("No answer to predict")

        input_ids = np.array(tmp_input_ids, dtype=np.int32)
        input_id_subs = np.array(tmp_input_sub, dtype=np.int32)
        context = np.full_like(tmp_input_ids, 1, dtype=np.int8)
        segment_ids = np.array(tmp_input_seg, dtype=np.int32)
        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)
        image_bound = np.array(image_bound, dtype=np.int32)

        for i, sample in enumerate(in_context_samples):# []/.不走这里
            (
                sample_input_ids,
                sample_id_subs,
                _,
                sample_segments,
                sample_rel,
                n_segments,
                table_states,
                image_bound,
            ) = convert_data_to_id(self.tokenizer, sample, table_states, max_depth=8)
            input_ids = np.concatenate([input_ids, sample_input_ids], axis=0)
            input_id_subs = np.concatenate([input_id_subs, sample_id_subs], axis=0)
            context = np.concatenate(
                [context, np.ones(sample_input_ids.shape, dtype=np.int8)], axis=0
            )
            segment_ids = np.concatenate([segment_ids, sample_segments], axis=0)
            segment_rel_offset = np.concatenate(
                [
                    segment_rel_offset,
                    np.full(sample_input_ids.shape, segment_rel.shape[0], dtype=np.int32),
                ],
                axis=0,
            )
            segment_rel = np.concatenate([segment_rel, sample_rel], axis=0)
            sample_ids = np.concatenate(
                [sample_ids, np.full(sample_input_ids.shape, i + 1, dtype=np.int32)], axis=0
            )
            num_segments = np.concatenate(
                [num_segments, np.full(sample_input_ids.shape, n_segments, dtype=np.int32)], axis=0
            )
        input_pos = np.arange(input_ids.shape[0], dtype=np.int32)

        return (
            input_ids,                  # 去掉<ans_1>对应的segment,剩下的8个segment的tokens
            input_id_subs,
            input_pos,
            context,                   # 同样去掉了最后一个seg.全1了。 108个
            segment_ids,               # 去掉了最后一个答案seg       
            #  segment_ids:[0,0,0, 1,1,1  2,2,2,2,2,2,2,2  3 3 3    4 4  5 5 5       6 6....    7,7,7]   只剩108个
            #         对应 [root,  image, <unk><unk><unk>, 'context',"", "question", "如果用...", <ans>]
            #                            64个token.这段留着，后续用img_embed填充                            
            #                           image_bound：[7,71],去掉头尾的真实img填充的64个tokens
            #  num_segments:[9 9  9 9 9 9...                                                        9,9]
            segment_rel_offset,        # 108个  全0
            segment_rel,               # 没变，还是81个。最全的seg间的关系。
            sample_ids,                # 108个 全0
            num_segments,              # 108个 剩下的tokens，每个值都是9.共9个segment.
            predict_segments,          # 放结果的segment,单独拆出来。segment 8，和对应的[(8,1)]  segment，和要预测的第一个ans
            answer_placeholders,       # [[]]
            table_states["ext_table"],      # {86583: '<ans_1>'}
            table_states["token_id_table"], # {'<ans>': {86583: 0}}
            image_bound                # [7,71],image_seg中，去掉头尾的，需要用真实img_embed填充的64个tokens. 所在的位置
        )

    def _process_list(self, data_list: List[Any], max_inp_length: Optional[int] = None):
        pack_tensor = []
        other_info = []
        segment_rel_pack = []

        batch_ext_table_map: Dict[Tuple[int, int], int] = {}
        batch_ext_table_ids: List[int] = []
        batch_ext_table_sub: List[int] = []

        for data in data_list:
            (
                input_ids,     # 每个seg对应的tokens，编码后，都拼到一起。 含img对应的<unk>，ques标签，编码后的内容
                input_id_subs, # 全0.和input_ids等长
                input_pos,     # 单纯是input_ids的位置， int32 (0,T-1)
                context,       # 去掉了最后一个seg.全1了。 108个. (原来有0是最后一个segment是预测答案对应的segment。)
                segment_ids,   
                segment_rel_offset, # 108个  全0
                segment_rel,        # 还是81个。9个seg间，22之间在树上的关系。
                sample_ids,         # 108个 全0
                num_segments,       # 108个，和input_ids等长。每个值是9，是总的9个segment
                predict_segments,   # 放结果的segment 8,单独拆出来。[(8,1)]
                answer_placeholders,# [[]]
                ext_table,          # {86583: '<ans_1>'}
                token_id_table,     # {'<ans>': {86583: 0}}
                image_bound         # [7,71],image_seg中，去掉头尾的，需要用真实img_embed填充的64个tokens. 所在的位置
            ) = self._convert_to_tensors(data, [], max_inp_length)
            
            #  input_ids：去掉<ans_1>对应的segment,剩下的8个segment的tokens
            #  input_pos： [0,1,2,...                                                         106,107]
            #  segment_ids:[0,0,0, 1,1,1  2,2,2,2,2,2,2,2  3 3 3    4 4  5 5 5       6 6....    7,7,7]   只剩108个
            #         对应 [root,  image, <unk><unk><unk>, 'context',"", "question", "如果用...", <ans>]
            #                            64个token.这段留着，后续用img_embed填充                            
            #                            image_bound：[7,71],去掉头尾的真实img填充的64个tokens
            #  num_segments:[9 9  9 9 9 9...                                                        9,9]
            
            # 把上述内容，都转成tensor (cuda),放进pad里
            rev_ext_table: Dict[int, str] = {}
            for token, mp in token_id_table.items():
                if token == "<ans>":
                    continue
                token_id = self.tokenizer.encoder[token]
                for fake_id, token_sub in mp.items():
                    if token_sub > 0:
                        if (token_id, token_sub) not in batch_ext_table_map:
                            batch_ext_table_map[(token_id, token_sub)] = (
                                len(batch_ext_table_ids) + self.tokenizer.vocab_size
                            )
                            batch_ext_table_ids.append(token_id)
                            batch_ext_table_sub.append(token_sub)
                        rev_ext_table[batch_ext_table_map[(token_id, token_sub)]] = ext_table[
                            fake_id
                        ]
                    else:
                        rev_ext_table[token_id] = ext_table[fake_id]
            # 输入的tokens，和所在的segment的信息，转成tensor            
            pack_tensor.append(
                {
                    "input_ids": torch.from_numpy(input_ids).unsqueeze(0),           # 内容不变。加了一维B=1。转成tensor(cuda)
                    "input_id_subs": torch.from_numpy(input_id_subs).unsqueeze(0),
                    "input_pos": torch.from_numpy(input_pos).unsqueeze(0),
                    "context": torch.from_numpy(context).unsqueeze(0),
                    "sample_idx": torch.from_numpy(sample_ids).unsqueeze(0),
                    "num_segments": torch.from_numpy(num_segments).unsqueeze(0),
                    "segment_ids": torch.from_numpy(segment_ids).unsqueeze(0),
                    "segment_rel_offset": torch.from_numpy(segment_rel_offset).unsqueeze(0),
                }
            )
            segment_rel_pack.append(torch.from_numpy(segment_rel))
            # 待预测的segment相关信息。，没有转tensor
            other_info.append(
                {
                    "predict_segments": predict_segments,                       
                    "answer_placeholders": answer_placeholders,
                    "ext_table": rev_ext_table,
                }
            )

        keys = set(pack_tensor[0].keys())
        padded = {}
        for key in keys:
            padded[key] = pad(pack_tensor, key).to(self.device)

        # 一些要放到padded中的内容，也都放torch tensor上。用来后续做输入。
        max_num_rels = 0
        for rel in segment_rel_pack:
            max_num_rels = max(max_num_rels, rel.size(0))
        padded_rels = torch.zeros(len(segment_rel_pack), max_num_rels, dtype=torch.int32)
        for i, rel in enumerate(segment_rel_pack):
            padded_rels[i, : rel.size(0)] = rel
        padded["segment_rel"] = padded_rels.to(self.device)     # 不变。各个seg之间的关系。torch.Size([1, 81])
        padded["batch_ext_table_ids"] = torch.tensor(           # []
            batch_ext_table_ids, dtype=torch.int32, device=self.device
        )
        padded["batch_ext_table_sub"] = torch.tensor(           # []
            batch_ext_table_sub, dtype=torch.int32, device=self.device
        )
        padded['image_bound'] = torch.from_numpy(image_bound).unsqueeze(0).to(self.device) # tensor([[[ 7, 71]]], device='cuda:0', dtype=torch.int32)
        return padded, other_info

    def generate(self, img_list, max_inp_length: Optional[int] = None, extra_inp_dict: dict = None, vision_hidden_states=None, return_vision_hidden_states=False, **kwargs):
        
        data_list = []     # 包含每张图片的： {image,context,question,<ans>}

        pixel_values = []  # 处理后的图片
        for img in img_list: # 原始图像。
            if vision_hidden_states is None:
                pixel_values.append(self.transform(img)) # 单张图像，torch.Size([3, 224, 224])  torch.float32
            inp_dict = {'image': self.tokenizer.unk_token * self.model.query_num} # 先填充成unk
            if extra_inp_dict:
                inp_dict.update(extra_inp_dict)
            inp_dict['<ans>'] = ''
            data_list.append(inp_dict)
            
        # data_list
        # [{'image': '<unk><unk><unk><unk>...<unk><unk>', 
        #   'context': '',
        #   'question': '如果用一句中国唐代的著名诗人"李白"的古...图像，你能想到什么？',
        #   '<ans>': ''    答案最后写回这里
        #  }
        # ]
        #
        # pixel_values:处理好的图片 [torch.Size([3, 224, 224])]

        # model_inputs中，输入data,都转成tensor了
        # dict_keys(['input_ids',    # torch.Size([1, 108]),          int32    含所有tokens编码后的结果。共8个segment对应的编码后id
        #            'input_pos',    # torch.Size([1, 108]),从0到107   int32
        #            'input_id_subs',# 全0
        #            'sample_idx',   # 全0
        #            'context',      # 全1
        #            'segment_ids',
        #            'num_segments', # 全9. 共9个segment，但最后一个不放input_ids里。对应答案token
        #            'segment_rel',  # torch.Size([1, 81])。9个seg间，22之间在树上的关系(节点深度)。
        #            'segment_rel_offset',# 全0  
        #            'batch_ext_table_ids', 
        #            'batch_ext_table_sub', 
        #            'image_bound']) # tensor([[[ 7, 71]]] 。去掉头尾的真实img填充的64个tokens，在input_tokens中的位置
        # 其中input_ids等tokens,都已经放到tensor(cuda)上了，作为后续输入
        #  segment_ids:[0,0,0, 1,1,1  2,2,2,2,2,2,2,2  3 3 3    4 4  5 5 5       6 6....    7,7,7]   只剩108个
        #         对应 [root,  image, <unk><unk><unk>, 'context',"", "question", "如果用...", <ans>]
        #                            64个token.这段留着，后续用img_embed填充                            
        #                            image_bound：[7,71],去掉头尾的真实img填充的64个tokens
        model_inputs, other_info = self._process_list(data_list, max_inp_length) # 不用管

        with torch.inference_mode():
            # 首次，传原始图片对应tensor(3,224,224).否则传已有的vision_embed
            if vision_hidden_states is None:
                pixel_values = torch.stack(pixel_values).to(self.device)
                model_inputs['pixel_values'] = pixel_values
            else:
                model_inputs['vision_hidden_states'] = vision_hidden_states


            # model_inputs['hidden_states']: (B,T,d) 原始文本tokens.embed+rotary embed后(B,T,d)。 其中rotary embed只针对特殊token
            #               占位的64个image token,用image_embed填充
            #               整体组合了文本，图像信息，作为llm模型decoder的初始输入
            # vision_hidden_states： 图像传入预训练的BEiT3，生成的image_embed （B,64,d）. 用64个tokens来表示。
            #                        返回，不单独用。多轮对话中重复用，image embed不变
            model_inputs['hidden_states'], vision_hidden_states = self.model.get_vllm_embedding(model_inputs)
            
            # 核心的infer逻辑。
            # 如果是首次，给定图片对应的tensor:       model_inputs['pixel_values']  torch.Size([3, 224, 224])  torch.float32
            #           否则给定上次生成的图片embed: model_inputs['vision_hidden_states']
            # result_ids:ans对应的tokens序列: [(word_id,1),(word_id,1),...]
            result_ids = self._decode(model_inputs, other_info, **kwargs)  # infer
        
        # 把ans对应的token_id，解析为文本
        for sent_id, result in enumerate(result_ids):
            
            # ans_result_map: {1: [177, 30545, 309]}  答案1对应的序列
            ans_result_map: Dict[int, List[int]] = {}
            for raw_word_id, ans_id in result:
                if ans_id not in ans_result_map:
                    ans_result_map[ans_id] = []
                ans_result_map[ans_id].append(raw_word_id)   # 首个ans对应的答案tokens. ans_result_map[1]:[word1,word2]

            answer_placeholders = other_info[sent_id]["answer_placeholders"]
            ext_table = other_info[sent_id]["ext_table"]
            data = data_list[sent_id]
            # 解析每个token_id,为word_text.  token_ids是答案序列
            for ans_id, token_ids in ans_result_map.items():  
                if token_ids[-1] == self.tokenizer.eos_id:
                    token_ids = token_ids[:-1]
                text = self.tokenizer.decode(token_ids, ext_table) # 整体解析出原文
                path = answer_placeholders[ans_id - 1]

                if len(path) > 0:
                    p = data["<ans>"]
                    for part in path[:-1]:
                        p = p[part]
                    p[path[-1]] = text
                else:
                    data["<ans>"] = text                         # 直接设置为答案
            for ans_id in range(len(answer_placeholders)):
                if (ans_id + 1) not in ans_result_map:
                    path = answer_placeholders[ans_id]
                    p = data["<ans>"]                       
                    for part in path[:-1]:
                        p = p[part]
                    p[path[-1]] = None

        if return_vision_hidden_states:
            # data_list：
            # [{'image': '<unk><unk>',
            #   'context': '', 
            #   'question': '如果用一句中国唐代的著名诗人"李白"的古诗来描述这幅图像，你能想到什么？', 
            #   '<ans>': '“黄河之水天上来，奔流到海不复回。” 李白的这句诗可以用来形容这幅图片中的景象：一条汹涌澎湃、波涛汹涌的河流从天而降，撞击着岩石峭壁，形成了令人叹为观止的壮观场面。'}]
            return data_list, vision_hidden_states   # 视觉embed固定.返回.下轮对话直接用。

        return data_list    # data_list[0]["<ans>"]，是文本形式额度答案了 。加速后，可以直接看这里的rouge_score/ 或者

    def _decode(self, model_inputs, other_info, **kwargs):
        raise NotImplementedError("_decode is not implemented.")


class VLLMCPMBeeBeamSearch(VLLMCPMBeeGeneration):
    """ use case
        beam_search = VLLMCPMBeeGeneration(vlu_cpmbee, tokenizer, transform)
        img = Image.open('xxx.jpg').convert('RGB')
        print(beam_search.generate([img], max_inp_length=128)[0]['<ans>'])
    """

    def _decode(
        self,
        model_inputs,  # 整体组合了文本，图像信息，作为llm模型decoder的初始输入。 （B,T,d）  单条ques的完整信息(含image内容)。
        other_info,
        beam_size=3,
        max_length=100,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
        repetition_window=None,
    ):
        """
        Beam search
        Args:
            model_inputs (dict): input ids. （1,T,d）
            beam_size (int, optional, defaults to 3): beam size of beam search.
            generate_length (int, optional, defaults to 100): maximum generation length.
            repetition_penalty (float, optional, defaults to 1.0): repetition penalty coefficient, 1.0 means no penalty.
            repetition_window (int, optional, defaults to None): window size of repetition penalty, None means that all output tokens are penalized.
        """  # noqa: E501
        # generate_length + 1 for EOS token. 多一个结束符
        max_length += 1

        # expand dimmension
        batch_size = model_inputs["input_ids"].size(0)  # 1
        
        # 原始tokens序列（1，T）（编码后的）->  中间加一维(B,1,T) -> expand成3个 -> (1,beam_size,T)
        # 转成（1,beam_size,T）的序列。
        # 相当于有3条输入序列。用来和后续3个不同token组合. 每次从新预测出的token中，选top3,再保留3条新的序列。
        # 为维护3条序列，额外开辟了连续内存，做beam_search
        input: torch.Tensor = (
            model_inputs["input_ids"]         # 原来已经是tensor了，对应所有segment(除<ans>之外的segment)的tokens (1,T)
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        # sub,pos等序列类似。最开始维护3条序列。等预测出新的token,再选出top3,维护3条新的序列。
        # 直到最后一个step，输出的token结束。从总共维护的3条序列里，选最好的一条。
        input_sub: torch.Tensor = (
            model_inputs["input_id_subs"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        input_pos: torch.Tensor = (
            model_inputs["input_pos"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        context: torch.Tensor = (
            model_inputs["context"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        sample_ids: torch.Tensor = (
            model_inputs["sample_idx"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        num_segments: torch.Tensor = (
            model_inputs["num_segments"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        segment: torch.Tensor = (
            model_inputs["segment_ids"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        segment_rel_offset: torch.Tensor = (
            model_inputs["segment_rel_offset"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        segment_rel: torch.Tensor = (
            model_inputs["segment_rel"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        # 甚至连初始输入序列对应的embed（B,T,d）,也维护了3份
        # （3B,T,d）
        hidden_states: torch.Tensor = (
            model_inputs["hidden_states"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, *model_inputs["hidden_states"].shape[1:]) # (3B=3,T=108,d=4096)
            .contiguous()
            .view(batch_size * beam_size, *model_inputs["hidden_states"].shape[1:])  # (3,108,4096)
        )
        ext_table_ids: torch.Tensor = model_inputs["batch_ext_table_ids"]  # [] int32
        ext_table_sub: torch.Tensor = model_inputs["batch_ext_table_sub"]  # []
        ext_table_ids_cpu = ext_table_ids.cpu()
        ext_table_sub_cpu = ext_table_sub.cpu()

        done = [False for _ in range(batch_size)]  # 如果多条一起，看batch中每个样本是否结束

        # 记录当前的3个beam序列,对应的分数 （1，3）   (1,beam_size)
        # 首个序列，初始分数是0，后2条序列，初始分数置为-inf
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=input.device)
        beam_scores[:, 1:] = -1e9             # 首个序列，初始分数是0，后2条序列，初始分数置为-inf
        beam_scores = beam_scores.view(-1)    # torch.Size([3])

        # generated hypotheses. beam_search的工具类
        # 这个池子永远只维护3个序列。含该序列生成完的ans,score
        # 每次已经生成完的一条序列（到了结束符），才加入到这个池子中。
        # 加入其中的序列，不影响剩下的3个beam。剩下的其他beam继续组成新序列，生成。直到也各自产出结束符，加入该池子。按分数，顶替已有序列
        # 最终生成完，留下
        generated_hyps = [
            # batch中每个样本（batch大小是1）,维护一个BeamHypotheses，里边是该beam序列的状态
            BeamHypotheses(beam_size, max_length, length_penalty=length_penalty, early_stopping=False)
            for _ in range(batch_size)
        ]
        
        

        pred_start_index = input.size(-1)  # T. 初始输入序列q的长度。
        
        # 首次构建。context阶段
        #   传入的序列只包含query。没有KVcache。past_key_values是None.
        #   传入3个相同的原始序列（和对应状态），作为beam_size初始的3条序列
        # past_key_values（dict）：
        #   buffer: 每层计算att时，用到的映射过的完整K矩阵,V矩阵。(B,T,d）
        #           首次，是每层映射过的K(B,Tq,d)，V(B,Tq,d).保存下来，后续用. 在本次计算时，直接用Q和该K,V相乘。（这里B==beam_size==3）
        #   其他：是首次计算时，原始q序列对应的状态。buffer_position，buffer_segments，都是原始q序列的一些信息（beam_size,Tq）
        _, _, past_key_values = \
        self.model.llm.inference( # 核心。CPMBeeTorch的infer方法
            input=input,          # （beam_size,T）的序列。相当于有3条输入序列。各自用来预测 next token.
            input_sub=input_sub,  # 都是3份，对应3个序列各自的状态. 对有下划线的特殊token做标记。正常token是0
            position=input_pos,   #  (beam_size,T）  首次输入，每条序列是query中各个token的pos。[0,1,2,...107] 
            context=context,      # 全1
            
            sample_ids=sample_ids,    #  (beam_size,T）全0
            num_segments=num_segments,#  (beam_size,T）全9
            segment=segment,          #  (beam_size,T）每条：是该序列输入token对应的segment
            segment_rel_offset=segment_rel_offset,#  (beam_size,T）全0
            segment_rel=segment_rel,              #  (beam_size,n_seg^2）  每条，对应每个seg和其他seg之间的关系。9个seg,81个关系。
            ext_table_ids=ext_table_ids,# []
            ext_table_sub=ext_table_sub,# []

            past_key_values=None,       # 构建阶段。没有KV cache, KV cache初始传入None
            
            hidden_states=hidden_states # (beam_size,T,d)   3个输入序列的embed.在此基础上，各自预测下一个token  (B=1)
        )

        beam_states = []
        for sent_id in range(batch_size):
            instance_beam_states = []        

            for beam_id in range(beam_size):
                instance_beam_states.append(      # 维护每个序列的3个beam状态
                    {
                        "idx": 0,                 
                        "ans": [],                # 维护该beam序列，截止目前生成的所有新的token,作为ans。[(word_id,1),(word_id,1)]
                        "nx_token_id": self.tokenizer.bos_id,
                        "nx_token_sub": 0,
                        "nx_segment_id": other_info[sent_id]["predict_segments"][0][0],  # 8. 新token,都属于答案segment
                        "nx_position": 0,
                    }
                )
            # 初始：
            #     每个序列，加一个begin_token。基于此做预测
            # 后续：
            #     上一次所有beam选出来的新的3个token,及其状态，维护在这里
            beam_states.append(instance_beam_states)  
            
        # gererate阶段
        # 每个step，新生成一个token
        for i in range(max_length + 1):
            # 每个step,确定给3个序列追加的一个token。之前维护在beam_state中
            tmp_input = []         # 3个token.追加到本次3条beam序列中
            tmp_input_sub = []     # 这3个token对应的sub
            tmp_position = []      # 这3个token对应的position
            tmp_segment = []       # 这3个token所属的segment
            for sent_id in range(batch_size):
                for beam_id in range(beam_size):
                    # 上一个step,所有beam选出来的新的3个token,及其状态.
                    tmp_input.append(beam_states[sent_id][beam_id]["nx_token_id"])      # [beam_size] .每个beam序列,加的一个token      
                    tmp_input_sub.append(beam_states[sent_id][beam_id]["nx_token_sub"])
                    tmp_position.append(beam_states[sent_id][beam_id]["nx_position"])
                    tmp_segment.append(beam_states[sent_id][beam_id]["nx_segment_id"])  # 该token的一些状态
            
            # 每个step的核心逻辑：
            with torch.no_grad():
                # 每个旧序列(beam_size,T）(原始的/beam_saerch选定的)，加上预测出来的新token. 作为新的beam序列。
                #  new_beam1=old_beam1  + token1
                #  new_beam2=old_beam1  + token2
                #  new_beam3=old_beam2  + token3
                input = torch.cat(   # （beam_size,T+1）  新的beam序列。每个beam_search后选中的旧序列，加上对应的新token
                    [
                        input,   # 上个step，经过beam_search,选定的旧序列
                        torch.tensor(tmp_input, dtype=torch.int32, device=self.device).view(
                            batch_size * beam_size, 1
                        ),       # 上个step,经过beam_search,得到的新token
                    ],
                    dim=-1,
                )
                
                # 一个step:
                # input：只包含一个新的token. 以及相关segment.q只有这一个token
                # past_key_values:传上次输入序列对应的状态。还不包含这个新token。和这个新token的状态拼一起，作为新的k，v序列（对应状态）
                # 计算时,q只用这个新token，得到
                #     attention_mask,   (B,tq=1,tk)
                #     position_bias     (B,heads,Tq=1,Tv)
                #     hidden_states     (B,1,d)
                # 而present_buffer中：
                #     是上次每层计算att时，用过的完整K矩阵,V矩阵。(B,T,d），不包含本次这个新token
                # 只计算该新token,对所有k,V的attention
                #     其中K,V是截止当前的所有token,含原始token和截止该step已经输出的所有token.不包含本次作为q的这个新token。
                #     该token还需要重新映射，计算q，k,v。
                #     已经缓存了的K,V大矩阵，拼上该token新的k,v向量，得到更新的大K,V矩阵，含q自己映射的向量（缓存下来，用作下次输入）
                #     计算该q对整个大K,V的attention，得到最终的h:(B，1，d),聚合了q位置对当前完整序列的att信息。
                #     输出logits(B,1,V),用来预测下一个token
                # past_key_values：
                #     截止当前的完整序列（含query和预测出的n个token），本次使用的KV矩阵。下次拼上新token对应的kv向量后，可以直接用
                logits, _, past_key_values = self.model.llm.inference(  # 核心。CPMBeeTorch的infer方法
                    # 当前序列的最后一个token。[beam_size,1]                                                  
                    input=input[:, -1:],  
                    # 该token对应的状态：pos,segment。[beam_size,1]
                    input_sub=torch.tensor(tmp_input_sub, dtype=torch.int32, device=self.device).view(# (beam_size,1)
                        batch_size * beam_size, 1
                    ),
                    position=torch.tensor(tmp_position, dtype=torch.int32, device=self.device).view(  # (beam_size,1)
                        batch_size * beam_size, 1
                    ),
                    context=torch.ones(
                        batch_size * beam_size, dtype=torch.bool, device=self.device
                    ).view(batch_size * beam_size, 1),# (beam_size,1)
                    sample_ids=torch.zeros(
                        batch_size * beam_size, dtype=torch.int32, device=self.device
                    ).view(batch_size * beam_size, 1),# (beam_size,1)
                    num_segments=num_segments[:, -1:],# (beam_size,1).值仍是9,总共9个segment
                    segment=torch.tensor(tmp_segment, dtype=torch.int32, device=self.device).view(
                        batch_size * beam_size, 1
                    ),                                # (beam_size,1)
                    segment_rel_offset=segment_rel_offset[:, -1:],
                    segment_rel=segment_rel,
                    ext_table_ids=ext_table_ids,
                    ext_table_sub=ext_table_sub,
                    # 上次输入序列的状态。还不包含本次这个新token。和本次拼起来，作为kv,被q att。
                    # 其中past_key_values['buffer'],是上次每层计算att时，用过的完整K矩阵,V矩阵。(B,T,d），还不包含本次这个新token
                    # 每层KV矩阵(B,T,d)，已经按照选中的老序列beam_id，抽取，替换成新的beam序列对应的状态了。（不含新token）
                    past_key_values=past_key_values, 
                )
                
                logits = logits[:, -1, :]  # （beam_size,V）  用来预测各序列的下一个token

            # skip all steps when we are done with each sentence
            if all(done):
                break

            for sent_id in range(batch_size):
                # 干掉logits里的<unk>
                # logits（beam_size,V）中， V中<unk>对应位置置位为-1000。
                if self.tokenizer.unk_id not in other_info[sent_id]["ext_table"]:
                    # unk is not allowed, mask unk
                    logits[
                        sent_id * beam_size: (sent_id + 1) * beam_size, self.tokenizer.unk_id
                    ] = -10000
                # 不在拓展词表里的词，对应位置也干掉
                ext_ids = set()
                for v in other_info[sent_id]["ext_table"].keys():
                    ext_ids.add(v)   # 加入词表
                for ext_id in range(
                    self.tokenizer.vocab_size, self.tokenizer.vocab_size + ext_table_ids.size(0)
                ):
                    if ext_id not in ext_ids:
                        logits[sent_id * beam_size: (sent_id + 1) * beam_size, ext_id] = -10000

            # 对logits进行调整，加上对重复的惩罚。
            apply_repetition_penalty(
                logits,
                batch_size,
                beam_size,
                input,
                repetition_penalty,   # repetition_penalty == 1.0, so no effect here.
                pred_start_index,
                input.size(-1) - 1,
                repetition_window,
            )
            # 对logits进行调整，加上对温度的惩罚。这里默认是1，不用管
            logits = logits / temperature
            
            # 真正用logits,预测下一个token。 softmax+log,作为每个token的score
            # log_softmax,所以scores的值为负数
            scores = F.log_softmax(logits, dim=-1)   # (beam_size,V)   对beam中每个序列，所有新token本身的scores
            

            # 把所有tokens的score,加到原来score上: 
            # (beam_size,V)。当前每个beam序列，可以选V种token
            # 首次： 
            #    之前的beam_score是[0,-1e9,-1e9]
            #    加上后，第2，3行，基本也都是-1e9 因为首次3条序列一样。从第一条里选3个token，组成新beam
            #    分数相加,等同于计算概率 log(p_old*p_token) 
            next_scores = scores + beam_scores[:, None].expand_as(  # (beam_size,V)   分数相加等同于计算概率 log(p_old*p_token)                                               
                scores
            )  # (batch_size * beam_size, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(batch_size, -1)  # (batch_size, beam_size * vocab_size)
            
            # 每个beam,V种选择。共beam*V种组合。按分数，选2beam个最佳组合
            next_scores, next_words = torch.topk(      # 选出的最佳score,对应的分数本身(2beam),和在所有beam*V个组合中的位置
                next_scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            
            assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)
            next_beam_states = []

            for sent_id in range(batch_size):
                # if we are done with this sentence
                
                # 如果已经生成完了,或者这个step得到的最好的分数，都差于之前池子中最差的，就整体结束
                done[sent_id] = done[sent_id] or \
                                generated_hyps[sent_id].is_done(next_scores[sent_id].max().item(), i)  
                                # next_scores[sent_id].max().item():  这些组合中，最高的分数
                                # i:当前已经生成的token(除了begin_token)。首次是0
                if done[sent_id]:
                    next_beam_states.append(   # 结束的句子。每个序列最后加一个结束token (0?)
                        [
                            (
                                {
                                    "idx": 0,
                                    "ans": [],
                                    "nx_token_id": 0,
                                    "nx_token_sub": 0,
                                    "nx_segment_id": 0,
                                    "nx_position": 0,
                                },
                                0,
                                0,
                            )
                        ]
                        * beam_size
                    )  # pad the batch
                    continue

                # 没结束，准备下一个beam的内容：
                next_instance_beam_states = []                   # next sentence beam content
                # 遍历选出来的6个候选token(beam-token组合)
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                    # idx: 某最佳组合，在（beam_size*V）中个组合中的位置. /V,%V可得对应的老beam_id，即选择的老beam序列
                    # value：该组合对应的score

                    # get beam and word IDs
                    beam_id = torch.div(idx, scores.size(-1), rounding_mode="floor").item()  # 选中的之前的序列（老beam_id）
                    word_id = (idx % scores.size(-1)).item()                                 # 选中的新token_id
                    curr_info = beam_states[sent_id][beam_id]   # 该beam当前的信息
                    
                    # end of sentence, or next word
                    # 如果某个组合：新选出的token是结束符/序列长度已达上限
                    if (
                        word_id == self.tokenizer.eos_id    # 是结束符 7
                        and (curr_info["idx"] + 1 == len(other_info[sent_id]["predict_segments"])) # 只需要输出一个ans
                    ) or i == max_length:                   # 该step长度达到上限
                        # 结束了。
                        # 把当前beam_state序列（生成的所有tokens）,加上本次预测出来这个token,,写到generated_hyps中。作为一条备选
                        # 不加入新的3条里。新的3条继续训练。训练完的，再加进去，顶掉
                        generated_hyps[sent_id].add(
                            beam_states[sent_id][beam_id]["ans"]  # 维护每个ans对应的token。(word_id,1)
                            + [
                                (
                                    word_id,
                                    other_info[sent_id]["predict_segments"][curr_info["idx"]][1],
                                )
                            ],
                            value.item(),
                        )
                    elif word_id == self.tokenizer.eos_id:  # 只是本序列终止了。其他序列还没有结束。本序列作为新beam中一个序列
                        next_instance_beam_states.append(
                            (
                                {
                                    "idx": curr_info["idx"] + 1,
                                    "ans": curr_info["ans"]
                                    + [
                                        (
                                            word_id,
                                            other_info[sent_id]["predict_segments"][
                                                curr_info["idx"]
                                            ][1],
                                        )
                                    ],
                                    "nx_token_id": self.tokenizer.bos_id,   # 新token,直接是一个结束符.后续不再生成新的token了
                                    "nx_token_sub": 0,
                                    "nx_segment_id": other_info[sent_id]["predict_segments"][
                                        curr_info["idx"] + 1
                                    ][0],
                                    "nx_position": 0,
                                },
                                value.item(),
                                sent_id * beam_size + beam_id,
                            )
                        )

                    else:
                        raw_word_id = word_id
                        word_id_sub = 0
                        if word_id >= self.tokenizer.vocab_size:
                            word_id -= self.tokenizer.vocab_size
                            word_id_sub = int(ext_table_sub_cpu[word_id].item())  # 拓展词
                            word_id = int(ext_table_ids_cpu[word_id].item())

                        next_instance_beam_states.append(
                            (
                                {
                                    "idx": curr_info["idx"],   # 这些内容不变
                                    "ans": curr_info["ans"]    # 初始是[],加上新token对应的(word_id,1)  属于第一个答案
                                    + [
                                        (
                                            raw_word_id,
                                            other_info[sent_id]["predict_segments"][
                                                curr_info["idx"]  # 第一个答案segment.  predict_segments[0]:(8, 1)。 
                                            ][1],
                                        )
                                    ],
                                    "nx_token_id": word_id,     # 新token,对应的各种状态
                                    "nx_token_sub": word_id_sub,
                                    "nx_segment_id": curr_info["nx_segment_id"],  # 仍属于答案segment
                                    "nx_position": curr_info["nx_position"] + 1,  # 该token在答案中的pos,+1
                                },
                                value.item(),                                    # 当前新beam的score
                                sent_id * beam_size + beam_id, # 该新token,对应的老beam. 拼到该beam的序列中，组成新的下一条序列。
                            )
                        )

                    # 挑够beam_size==3个，就break。（其他的不管了。或者有的beam结束了。）
                    if len(next_instance_beam_states) == beam_size:
                        break

                # update next beam content
                assert len(next_instance_beam_states) == 0 if i == max_length else beam_size
                next_beam_states.append(next_instance_beam_states)

            # we have reached the last step
            if i == max_length:
                break

            # sanity check / prepare next batch
            # 构建新的beam,对应的状态
            beam_reorder_idx = []
            beam_new_scores = []                  # 新的beam对应的3个score (后续转为tensor)
            beam_states = []                      # 新的beam对应的3个token的状态。添加到已有的beam序列上，用来生成新的k,v。  
            for sent_id in range(batch_size):
                instance_beam_states = []
                for beam_id in range(beam_size):
                    state, value, beam_idx = next_beam_states[sent_id][beam_id]  # 每个beam,新的序列
                    beam_reorder_idx.append(beam_idx)    # 对应的老beam的id
                    beam_new_scores.append(value)        # 新beam的score
                    instance_beam_states.append(state)   # 该新token的state.用于下一次追加到K,V中
                beam_states.append(instance_beam_states)

            input = input[beam_reorder_idx, :]    # （beam_size,Told） 新beam中，要保留的老beam对应序列。
                                                  #  下个step,各自拼上新token,作为完整的当前beam序列
                                                  #  输入decoder时，q只用最后的新token。
            
            # 新的beam对应的3个score.后续得到新token的logits后，相应分数追加上去。
            beam_scores = torch.tensor(beam_new_scores, dtype=torch.float, device=input.device) # 新的beam对应的3个score.
                      
            # 更新旧序列相关的状态，包括KV cache。
            # 按老beam_id抽取留下的旧序列，作为3个新序列的前半部分                                          
            for kw in past_key_values.keys():
                if kw == "buffer":
                    buf_list = past_key_values[kw]  # 本次各个beam序列保存的KV矩阵. 每层是(B,T,d)
                    nw_buf_list = []
                    for k_buf, v_buf in buf_list:   # 每层的KV矩阵 （B,T,d）。按选中的老beam_id,抽取出新beam要保留的序列.重组
                        # 新的KV矩阵。仍是(B，T,d)
                        nw_buf_list.append((k_buf[beam_reorder_idx, :], v_buf[beam_reorder_idx, :])) # 仍是(beam_size,T,d)
                    past_key_values[kw] = nw_buf_list # 新的KV cache.每层的(B,T,d)，已经替换成新的beam序列对应的state了。
                else:
                    past_key_values[kw] = past_key_values[kw][beam_reorder_idx, :]  # 其他状态，都抽取出老beam对应状态

        # 达到最大长度，所有step结束
        # 从池子中，选择分数最大的序列，返回
        # select the best hypotheses
        results = []
        for sent_id, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]   # 从当前池子的3个最佳beam中，选一个score最大的。把tokens加入result
            results.append(best_hyp)                                # 返回ans对应的tokens序列: [(word1,1),(word2,1),...]
            
        # ans对应的tokens序列: [(word_id,1),(word_id,1),...]
        return results
