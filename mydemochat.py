
# 如果您单卡显存不足40g，可以引入如下环境变量并将安全模块开关关闭。引入后显存占用约为5G，但推理所需时间会变长。此选项依赖bminf，需要安装bminf依赖库。
# export CUDA_MEMORY_CPMBEE_MAX=1g

from VisCPM import VisCPMChat
from PIL import Image

model_path = 'pytorch_model.zhplus.bin'
# 默认关掉安全检查。不设置CUDA_MEMORY_CPMBEE_MAX，整个都放显存。此外没有其他加速
viscpm_chat = VisCPMChat(model_path, image_safety_checker=False) 

# 不开启对输入图片的安全检查
image_path = 'figures/vlu_case1.png'
image = Image.open(image_path).convert("RGB")

question = '如果用一句中国唐代的著名诗人"李白"的古诗来描述这幅图像，你能想到什么？'
answer, _, _ = viscpm_chat.chat(image, question)

print(answer)


# 多轮对话：
image_path = 'figures/vlu_case2.jpeg'
image = Image.open(image_path).convert("RGB")

question = '这幅图像是在哪个节日拍摄的？'
answer, context, vision_hidden_states = viscpm_chat.chat(image, question)

# 多轮对话， 传入历史 context（文本形式，含上次答案），以及上次生成的图像embed
question = '你能用什么古诗描述这幅画？'
answer, context, _ = viscpm_chat.chat(image, question, context, vision_hidden_states=vision_hidden_states)

print(context)
