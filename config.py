# Hugging Face API 配置
HF_API_TOKEN = "your_huggingface_token_here"  # 替换为你的Token

# 模型ID配置
TEXT_MODEL = "IDEA-CCNL/Wenzhong-GPT2-110M"  # 中文故事生成
IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"  # 插图生成

# 生成参数
STORY_SETTINGS = {"max_length": 300, "do_sample": True, "temperature": 0.7}

IMAGE_SETTINGS = {"height": 512, "width": 512, "num_inference_steps": 30}
