import os
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fairytale.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# API配置
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("请设置HF_API_TOKEN环境变量")

# 模型配置
TEXT_MODEL = "IDEA-CCNL/Wenzhong-GPT2-110M"
IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"

# 生成参数
STORY_SETTINGS = {"max_length": 300, "do_sample": True, "temperature": 0.7}

IMAGE_SETTINGS = {"height": 512, "width": 512, "num_inference_steps": 30}

