import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_CONFIG = {
    "name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "local_dir": str(BASE_DIR / "models/Qwen2-1.5B"),
    "generation": {
        "max_new_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.8,
    },
    "system_prompt": """你是一位专业童话作家...""",  # 保持原有内容
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

