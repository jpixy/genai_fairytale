import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent

# 使用 Qwen2-1.5B-Instruct 模型
MODEL_CONFIG = {
    "name": "Qwen/Qwen2-1.5B-Instruct",
    "local_dir": str(BASE_DIR / "models/Qwen2-1.5B"),
    "generation": {
        "max_new_tokens": 500,  # 生成长度
        "temperature": 0.7,  # 随机性
        "top_p": 0.9,  # 核心采样
        "repetition_penalty": 1.5,  # 重复惩罚
    },
    "system_prompt": """你是一位资深童话作家，擅长创作适合6-10岁儿童的童话故事。请严格遵循以下规则：
1. 字数：350-450字
2. 结构：
   [背景] 50字引入
   [冒险] 150字发展
   [高潮] 200字转折
   [结局] 50字收尾
   [寓意] 1句总结
3. 要求：
   - 使用3个以上拟声词
   - 包含1次对话
   - 避免负面内容""",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

