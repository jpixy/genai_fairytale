import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from config import logger, MODEL_CONFIG
import json


class StoryGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device type: {self.device.upper()}")
        self.model = None
        self.tokenizer = None
        self.model_dir = Path(MODEL_CONFIG["local_dir"])

    def _patch_model_config(self):
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r+", encoding="utf-8") as f:
                config = json.load(f)
                config["use_flash_attention_2"] = False
                config["_attn_implementation"] = "eager"
                if "attn_config" in config:
                    config["attn_config"]["attn_impl"] = "eager"
                f.seek(0)
                json.dump(config, f, indent=2)
                f.truncate()
                logger.info("Model config patched")

    def load_model(self):
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)

            if not any(self.model_dir.glob("*")):
                logger.info("Downloading model")
                snapshot_download(
                    repo_id=MODEL_CONFIG["name"],
                    local_dir=self.model_dir,
                    ignore_patterns=["*.bin"],
                    resume_download=True,
                )

            self._patch_model_config()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            ).eval()

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def generate(self, keyword):
        prompt = f"请用500字创作关于【{keyword}】的童话故事，要求包含开头、发展、高潮、结局四部分，语言生动适合儿童阅读。"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return story.split("】的童话故事，")[-1].strip()

