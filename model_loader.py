import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import logging
import time
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class StoryGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.model_dir = Path("models/Qwen2-VL-2B")

        # 检查CUDA兼容性
        if self.device == "cuda":
            self._check_cuda_compatibility()

    def _check_cuda_compatibility(self):
        """检查CUDA和PyTorch版本兼容性"""
        cuda_version = torch.version.cuda
        logger.info(f"CUDA Version: {cuda_version}")
        logger.info(f"PyTorch Version: {torch.__version__}")

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

    def _download_model(self):
        """处理模型下载"""
        try:
            # 检查磁盘空间
            total, used, free = shutil.disk_usage("/")
            logger.info(
                f"Disk Space - Total: {total // (2**30)}GB, Free: {free // (2**30)}GB"
            )

            # 下载模型
            snapshot_download(
                repo_id="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                local_dir=self.model_dir,
                resume_download=True,
                ignore_patterns=["*.bin", "*.h5"],  # 忽略不必要的文件
                max_workers=4,
                token=None,
            )
            return True
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            return False

    def _verify_files(self):
        """验证下载的文件"""
        required_files = [
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]

        for file in required_files:
            if not (self.model_dir / file).exists():
                logger.error(f"Missing file: {file}")
                return False
        return True

    def load_model(self):
        """加载模型主流程"""
        try:
            # 准备目录
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # 下载检查
            if not any(self.model_dir.glob("*.safetensors*")):
                logger.info("Starting download...")
                if not self._download_model() or not self._verify_files():
                    raise RuntimeError("Download failed")

            # 加载tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).eval()

            # 测试生成
            logger.info("Testing generation...")
            test_output = self.generate("测试")
            logger.info(f"Test output length: {len(test_output)}")

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Load failed: {str(e)}")
            if "CUDA out of memory" in str(e):
                logger.error("Try reducing max_new_tokens or using smaller model")
            raise

    def generate(self, keyword, max_length=700):
        """生成故事"""
        try:
            prompt = f"请用{max_length}字创作关于【{keyword}】的童话故事"

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            outputs = self.model.generate(
                **inputs, max_new_tokens=max_length, temperature=0.7, do_sample=True
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

