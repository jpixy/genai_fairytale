import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from config import logger, MODEL_CONFIG
import time
import shutil
import os


class StoryGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.model_dir = Path(MODEL_CONFIG["local_dir"])
        self._setup_logging()

    def _setup_logging(self):
        logger.info("=" * 60)
        logger.info(f"Initializing Story Generator")
        logger.info(f"Device: {self.device.upper()}")
        logger.info(f"Model: {MODEL_CONFIG['name']}")
        logger.info(f"Cache dir: {self.model_dir}")
        logger.info("=" * 60)

    def _download_model(self):
        logger.info("Starting model download...")

        try:
            # 检查磁盘空间
            total, used, free = shutil.disk_usage("/")
            logger.info(
                f"Disk space - Total: {total // (2**30)}GB, Free: {free // (2**30)}GB"
            )

            # 使用更简单的下载方式，避免自定义进度条的问题
            snapshot_download(
                repo_id=MODEL_CONFIG["name"],
                local_dir=self.model_dir,
                resume_download=True,
                local_dir_use_symlinks=False,
                token=None,
            )

            logger.info("Download completed successfully")
            return True

        except Exception as e:
            logger.error(f"Download failed: {str(e)}", exc_info=True)
            return False

    def _verify_download(self):
        """验证下载的文件完整性"""
        logger.info("Verifying downloaded files...")
        required_files = [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
        ]

        for file in required_files:
            if not (self.model_dir / file).exists():
                logger.error(f"Missing file: {file}")
                return False

        logger.info("All required files present")
        return True

    def load_model(self):
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model directory ready: {self.model_dir}")

            if not any(self.model_dir.glob("*.safetensors*")):
                logger.info("No model files found, starting download...")
                if not self._download_model():
                    raise RuntimeError("Model download failed")

                if not self._verify_download():
                    raise RuntimeError("Download verification failed")
            else:
                logger.info("Found existing model files")

            logger.info("Loading tokenizer...")
            load_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer loaded in {time.time() - load_start:.2f}s")

            logger.info("Loading model...")
            load_start = time.time()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).eval()
            logger.info(f"Model loaded in {time.time() - load_start:.2f}s")

            logger.info("Running test generation...")
            test_output = self.generate("测试")
            logger.info(
                f"Test generation successful (length: {len(test_output)} chars)"
            )

            logger.info("=" * 60)
            logger.info("Model ready for use")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error("!" * 60)
            logger.error("MODEL LOADING FAILED", exc_info=True)
            logger.error("!" * 60)
            raise

    def generate(self, keyword):
        try:
            logger.info(f"Starting generation for: {keyword}")
            prompt = f"请用700字创作关于【{keyword}】的童话故事，包含完整的故事结构，语言生动有趣。"

            token_start = time.time()
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self.device)
            logger.info(f"Tokenization completed in {time.time() - token_start:.2f}s")

            gen_start = time.time()
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
            logger.info(f"Generation completed in {time.time() - gen_start:.2f}s")

            decode_start = time.time()
            story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = story.split("】的童话故事，")[-1].strip()
            logger.info(f"Decoding completed in {time.time() - decode_start:.2f}s")
            logger.info(f"Generated story length: {len(result)} characters")

            return result

        except Exception as e:
            logger.error("Generation failed", exc_info=True)
            raise

