import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_CONFIG = {
    "name": "Alibaba-NLP/gte-Qwen2-3B-instruct",
    "local_dir": str(BASE_DIR / "models/Qwen2-3B"),
    "generation": {
        "max_new_tokens": 800,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.8,
    },
    "system_prompt": """你是一位专业童话作家，请创作700字左右的童话故事...""",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # 简化日志格式
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

