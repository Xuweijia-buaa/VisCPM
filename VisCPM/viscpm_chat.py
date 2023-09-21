import os
import numpy as np
import torch
from PIL import Image
from VisCPM.cpm_tokenizers import CPMBeeTokenizer
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from timm.models import create_model
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
from transformers import CLIPImageProcessor
from typing import List

from VisCPM.generation.vllm_bee import VLLMCPMBeeBeamSearch
from VisCPM.models import VLU_CPMBee
from VisCPM.models.cpmbee import CPMBeeConfig, CPMBeeTorch
from VisCPM.utils import utils
import bminf

file_path = os.path.dirname(__file__)


def grid_image(images: List[Image.Image]) -> Image.Image:
    n = len(images)
    nrow = min(n, 8)
    images_tensor = [to_tensor(image) for image in images]
    images_tensor_grid = make_grid(images_tensor, nrow, padding=0)
    images_grid = to_pil_image(images_tensor_grid)
    return images_grid


class VisCPMChat(object):
    def __init__(self, model_path, config_path=None, image_safety_checker=False) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = utils.build_transform(is_train=False)
        self.tokenizer = CPMBeeTokenizer()

       #  一个预训练的beit3模型（timm导入的）。多模态。Multiway Transformer
       # 输入图像，经过CNNEncoder，Transformer等编码
        self.beit3_wrapper = create_model("beit3_large_patch16_224")  # 是torch.nn.Module。包含一个beit3和一个mim_head
        if config_path is None:
            config_path = os.path.join(file_path, '../config/cpm-bee-10b.json')
        self.config = CPMBeeConfig.from_json_file(config_path)  # heads.layers等参数
        
        # CPM模型。是其中的llmmodel。是torch.nn.Module
        self.cpm_model = CPMBeeTorch(self.config)

       # 整个是一个torch.nn.Module
        self.vlu_cpmbee = VLU_CPMBee(                  
            llm=self.cpm_model,                        # 其中的llm模型
            vpm=self.beit3_wrapper,                    # 其中的多模态视觉model
            vision_dim=self.beit3_wrapper.args.encoder_embed_dim,# 和对应的维度
            query_num=64,
            device=self.device
        )

        # 把上述结构，封装到beam_search中。infe时,用到self.vlu_cpmbee
        self.beam_search = VLLMCPMBeeBeamSearch(  
            self.vlu_cpmbee, self.tokenizer, self.transform, device=self.device
        )

        # 加载原始的模型参数，给self.vlu_cpmbee
        # 该模型。之后替换成推理。对应的模型权重，拿给engine
        vlu_state_dict = torch.load(model_path, map_location="cpu")
        self.vlu_cpmbee.load_state_dict(vlu_state_dict)  
        
        # 用了torch.nn.Module.half.就地把floating参数，都改成了half (half datatype)
        self.vlu_cpmbee.half()                           # 模型原来直接用. 问下老师这个，我们是否要保持。还有他原来推理用了一个bminf的包

        if os.getenv('CUDA_MEMORY_CPMBEE_MAX', False):   # 默认不设置.不用bminf。
            limit = os.getenv("CUDA_MEMORY_CPMBEE_MAX")
            try:
                assert limit.lower().endswith('g')
                memory_limit = int(limit.lower()[:-1]) * (1 << 30)
                print(f'use CUDA_MEMORY_CPMBEE_MAX={limit} to limit cpmbee cuda memory cost ')
            except:
                memory_limit = None
                print(f'environment CUDA_MEMORY_CPMBEE_MAX={limit} parse error')

            self.cpm_model = bminf.wrapper(self.cpm_model, memory_limit=memory_limit)  # 原来用了一个bminf的包，把llm模型包装了一下
            self.vlu_cpmbee.query.data = self.vlu_cpmbee.query.data.to(self.device)
            self.vlu_cpmbee.mapping.to(self.device)
            self.vlu_cpmbee.vpm.to(self.device)   # 视觉模型，直接放显存
        else:
            self.vlu_cpmbee.to(self.device)       # 整个放显存

        self.vlu_cpmbee.eval()                      # 模型本身，是eval模式

        if image_safety_checker: # 默认关
            self.image_safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            )
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )  # Download image_processing_config from huggingface.co and cache.
            self.image_safety_checker.to(self.device)
        else:
            self.image_safety_checker = None
            self.feature_extractor = None

    def chat(self, image, question, context='', vision_hidden_states=None):
        extra_inp_dict = {
            "context": context,
            "question": question, # '如果用一句中国唐代的著名诗人"李白"的古诗来描述这幅图像，你能想到什么？'
        }

        images, has_nsfw_concept = self.run_image_safety_checker( # 不执行。返回None
            [np.asarray(image)], self.device, torch.float
        )
        if has_nsfw_concept and has_nsfw_concept[0]:
            print("Content is not safe for work.")
            images = grid_image(np.asarray(image))

        # 核心的推理逻辑
        res, vision_hidden_states = self.beam_search.generate(
            [image],                       # 我们上传的图片
            max_inp_length=3000,
            max_length=512,
            extra_inp_dict=extra_inp_dict,  # 包含我们的文字
            vision_hidden_states=vision_hidden_states, # 首次没有图像embed. 第二轮对话，用前一轮生成的
            return_vision_hidden_states=True,
            beam_size=3,
            temperature=0.7,
            repetition_penalty=1.1,
            length_penalty=3,
        )
        
        # 已经是文本形式的答案
        answer = res[0]["<ans>"]       

        # context： 分2行，一行是原始问题，一行是本次回答
        #           和生成的图像embed一起，用作下次chat的context。
        context += "User: " + question + "\n"
        context += "AI: " + answer + "\n"
        return answer, context, vision_hidden_states

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_image_safety_checker(self, image, device, dtype):
        if self.image_safety_checker is not None:
            image_safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(np.asarray(image)), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.image_safety_checker(
                images=image, clip_input=image_safety_checker_input.pixel_values.to(dtype)
            )
            if any(has_nsfw_concept):
                print(
                    "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                )
                for idx, _has_nsfw_concept in enumerate(has_nsfw_concept):
                    if _has_nsfw_concept:
                        image[idx] = np.zeros(image[idx].shape)  # black image
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

