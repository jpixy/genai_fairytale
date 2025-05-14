import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, HfApi
from config import logger, MODEL_CONFIG
import json
from tqdm.auto import tqdm
import time
import requests
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

    def _custom_tqdm(self, iterable=None, **kwargs):
        """Kaggle优化的TQDM进度条"""
        return tqdm(
            iterable,
            bar_format="{l_bar}{bar:20}{r_bar}",
            dynamic_ncols=True,
            mininterval=1,
            **kwargs,
        )

    def _download_model(self):
        logger.info("Starting model download...")

        try:
            # 检查磁盘空间
            total, used, free = shutil.disk_usage("/")
            logger.info(
                f"Disk space - Total: {total // (2**30)}GB, Used: {used // (2**30)}GB, Free: {free // (2**30)}GB"
            )

            # 创建自定义下载回调
            class DownloadProgress(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with DownloadProgress(
                unit="B",
                unit_scale=True,
                miniters=1,
                desc="Downloading",
                bar_format="{l_bar}{bar:20}{r_bar}",
            ) as t:
                snapshot_download(
                    repo_id=MODEL_CONFIG["name"],
                    local_dir=self.model_dir,
                    resume_download=True,
                    tqdm_class=lambda *args, **kwargs: t,
                    local_dir_use_symlinks=False,
                    token=os.getenv("HUGGINGFACE_TOKEN"),
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

        missing_files = []
        for file in required_files:
            if not (self.model_dir / file).exists():
                missing_files.append(file)

        if missing_files:
            logger.error(f"Missing critical files: {missing_files}")
            return False

        logger.info("All required files present")
        return True

    def load_model(self):
        try:
            # 步骤1: 创建模型目录
            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model directory ready: {self.model_dir}")

            # 步骤2: 检查是否已下载
            if not any(self.model_dir.glob("*.safetensors*")):
                logger.info("No model files found, starting download...")
                if not self._download_model():
                    raise RuntimeError("Model download failed")

                if not self._verify_download():
                    raise RuntimeError("Download verification failed")
            else:
                logger.info("Found existing model files")

            # 步骤3: 加载tokenizer
            logger.info("Loading tokenizer...")
            load_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer loaded in {time.time() - load_start:.2f}s")

            # 步骤4: 加载模型
            logger.info("Loading model...")
            load_start = time.time()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).eval()
            logger.info(f"Model loaded in {time.time() - load_start:.2f}s")

            # 步骤5: 验证模型
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

            # Tokenization
            token_start = time.time()
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self.device)
            logger.info(f"Tokenization completed in {time.time() - token_start:.2f}s")

            # Generation with progress
            gen_start = time.time()
            with self._custom_tqdm(
                total=MODEL_CONFIG["generation"]["max_new_tokens"],
                desc="Generating",
                unit="tokens",
            ) as pbar:

                def progress_callback(**kwargs):
                    pbar.update(1)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MODEL_CONFIG["generation"]["max_new_tokens"],
                    temperature=MODEL_CONFIG["generation"]["temperature"],
                    top_p=MODEL_CONFIG["generation"]["top_p"],
                    do_sample=MODEL_CONFIG["generation"]["do_sample"],
                    repetition_penalty=MODEL_CONFIG["generation"]["repetition_penalty"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    callback=progress_callback,
                )

            logger.info(f"Generation completed in {time.time() - gen_start:.2f}s")

            # Decoding
            decode_start = time.time()
            story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = story.split("】的童话故事，")[-1].strip()
            logger.info(f"Decoding completed in {time.time() - decode_start:.2f}s")
            logger.info(f"Generated story length: {len(result)} characters")

            return result

        except Exception as e:
            logger.error("Generation failed", exc_info=True)
            raise

