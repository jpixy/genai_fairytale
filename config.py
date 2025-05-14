import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_CONFIG = {
    "name": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    "local_dir": str(BASE_DIR / "models/Qwen2-VL-2B"),
    "generation": {
        "max_new_tokens": 900,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.8,
    },
    "system_prompt": """你是一位专业童话作家，请创作700字左右的童话故事...""",
    "max_retries": 3,
    "retry_delay": 5,
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)

