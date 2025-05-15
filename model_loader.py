import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import logging
import time
import shutil
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class StoryGenerator:
    def __init__(self):
        self._check_environment()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.model_dir = Path("models/Qwen2-VL-2B")

    def _check_environment(self):
        """检查关键依赖版本"""
        logger.info("Checking environment...")
        try:
            import transformers

            logger.info(f"Transformers version: {transformers.__version__}")

            if torch.cuda.is_available():
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available, using CPU")
        except Exception as e:
            logger.error(f"Environment check failed: {str(e)}")
            raise

    def _download_model(self, max_retries=3):
        """带重试机制的模型下载"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{max_retries}")

                # 清理不完整的下载
                if attempt > 0:
                    shutil.rmtree(self.model_dir, ignore_errors=True)
                    self.model_dir.mkdir(parents=True)

                snapshot_download(
                    repo_id="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                    local_dir=self.model_dir,
                    resume_download=True,
                    ignore_patterns=["*.bin", "*.h5", "*.ot"],
                    max_workers=4,
                    token=None,
                )
                return True
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return False
                time.sleep(5)

    def _verify_download(self):
        """验证下载完整性"""
        required_files = {
            "config.json": 0.01,
            "model.safetensors.index.json": 0.1,
            "model-00001-of-00003.safetensors": 1.5,
            "model-00002-of-00003.safetensors": 1.5,
            "model-00003-of-00003.safetensors": 1.0,
        }

        for filename, min_size_gb in required_files.items():
            filepath = self.model_dir / filename
            if not filepath.exists():
                logger.error(f"Missing file: {filename}")
                return False
            size_gb = filepath.stat().st_size / (1024**3)
            if size_gb < min_size_gb:
                logger.error(
                    f"File too small: {filename} ({size_gb:.2f}GB < {min_size_gb}GB)"
                )
                return False
        return True

    def load_model(self):
        """加载模型主流程"""
        try:
            # 准备目录
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # 下载检查
            if not any(self.model_dir.glob("*.safetensors*")):
                if not self._download_model() or not self._verify_download():
                    raise RuntimeError("Model download failed")

            # 加载tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型
            logger.info("Loading model (this may take several minutes)...")
            torch.cuda.empty_cache()  # 清理GPU缓存

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).eval()

            # 测试生成
            logger.info("Testing generation...")
            test_output = self.generate("测试", max_length=50)
            logger.info(f"Test output: {test_output[:100]}...")

            logger.info("Model loaded successfully")
            return True

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory! Try:")
            logger.error("1. Restart kernel and free memory")
            logger.error("2. Reduce model size")
            logger.error("3. Use smaller max_length")
            raise
        except Exception as e:
            logger.error(f"Load failed: {type(e).__name__}: {str(e)}")
            raise

    def generate(self, keyword, max_length=700):
        """生成故事"""
        try:
            prompt = f"请用{max_length}字创作关于【{keyword}】的童话故事"

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Generation failed: {type(e).__name__}: {str(e)}")
            raise

