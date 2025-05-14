import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, HfApi
from config import logger, MODEL_CONFIG
import json
from tqdm import tqdm
import time
import requests


class StoryGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.model_dir = Path(MODEL_CONFIG["local_dir"])
        self._setup_logging()

    def _setup_logging(self):
        logger.info(f"Initializing story generator on {self.device.upper()}")
        logger.info(f"Model: {MODEL_CONFIG['name']}")
        logger.info(f"Cache dir: {self.model_dir}")

    def _verify_model_access(self):
        try:
            api = HfApi()
            model_info = api.model_info(MODEL_CONFIG["name"])
            logger.info(
                f"Model found - Size: {model_info.safetensors['total'] / 1024 / 1024:.2f}MB"
            )
            return True
        except Exception as e:
            logger.error(f"Model access verification failed: {str(e)}")
            return False

    def _download_model(self):
        for attempt in range(MODEL_CONFIG["max_retries"]):
            try:
                logger.info(
                    f"Download attempt {attempt + 1}/{MODEL_CONFIG['max_retries']}"
                )

                snapshot_download(
                    repo_id=MODEL_CONFIG["name"],
                    local_dir=self.model_dir,
                    resume_download=True,
                    tqdm_class=tqdm,
                    local_dir_use_symlinks=False,
                    token=True,  # 使用HuggingFace token
                )
                return True

            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP Error: {http_err}")
                if http_err.response.status_code == 401:
                    logger.error(
                        "Authentication required. Please set HUGGINGFACE_TOKEN environment variable"
                    )
                time.sleep(MODEL_CONFIG["retry_delay"])
            except Exception as e:
                logger.error(f"Download failed: {str(e)}")
                time.sleep(MODEL_CONFIG["retry_delay"])

        return False

    def load_model(self):
        try:
            if not self._verify_model_access():
                raise RuntimeError("Model access verification failed")

            self.model_dir.mkdir(parents=True, exist_ok=True)

            if not any(self.model_dir.glob("*.safetensors")):
                if not self._download_model():
                    raise RuntimeError("Model download failed after retries")

            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).eval()

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}", exc_info=True)
            raise

    def generate(self, keyword):
        try:
            prompt = f"请用700字创作关于【{keyword}】的童话故事，包含完整的故事结构，语言生动有趣。"

            logger.info(f"Generating story for: {keyword}")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MODEL_CONFIG["generation"]["max_new_tokens"],
                temperature=MODEL_CONFIG["generation"]["temperature"],
                top_p=MODEL_CONFIG["generation"]["top_p"],
                do_sample=MODEL_CONFIG["generation"]["do_sample"],
                repetition_penalty=MODEL_CONFIG["generation"]["repetition_penalty"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return story.split("】的童话故事，")[-1].strip()

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise

