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
        """初始化日志记录"""
        logger.info("\n" + "=" * 80)
        logger.info("Initializing Story Generator")
        logger.info(f"Device: {self.device.upper()}")
        logger.info(f"Model: {MODEL_CONFIG['name']}")
        logger.info(f"Cache dir: {self.model_dir}")
        logger.info("=" * 80 + "\n")

    def _download_model(self):
        """处理模型下载流程"""
        logger.info("Starting model download...")

        try:
            # 显示磁盘空间
            total, used, free = shutil.disk_usage("/")
            logger.info(
                f"Disk space - Total: {total // (2**30)}GB, Free: {free // (2**30)}GB"
            )

            # 下载模型文件
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

    def _verify_model_files(self):
        """验证下载的文件完整性"""
        logger.info("Verifying downloaded files...")
        required_files = [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]

        missing_files = []
        for file in required_files:
            if not (self.model_dir / file).exists():
                missing_files.append(file)
                logger.error(f"Missing file: {file}")
            else:
                size = (self.model_dir / file).stat().st_size / (1024**3)
                logger.info(f"Verified: {file} ({size:.2f}GB)")

        return len(missing_files) == 0

    def _log_memory_usage(self):
        """记录内存使用情况"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(
                f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            )

    def load_model(self):
        """加载模型主流程"""
        try:
            # 阶段1: 准备目录
            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model directory ready: {self.model_dir}")

            # 阶段2: 下载检查
            if not any(self.model_dir.glob("*.safetensors*")):
                logger.info("No model files found, starting download...")
                if not self._download_model():
                    raise RuntimeError("Model download failed")

                if not self._verify_model_files():
                    raise RuntimeError("Download verification failed")
            else:
                logger.info("Found existing model files")
                if not self._verify_model_files():
                    logger.warning(
                        "Existing files are incomplete, cleaning and retrying..."
                    )
                    shutil.rmtree(self.model_dir)
                    return self.load_model()

            # 阶段3: 加载tokenizer
            logger.info("Loading tokenizer...")
            tokenizer_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(
                f"Tokenizer loaded in {time.time() - tokenizer_start:.2f} seconds"
            )

            # 阶段4: 加载模型
            logger.info("Loading model...")
            model_start = time.time()
            self._log_memory_usage()

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).eval()

            logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")
            self._log_memory_usage()

            # 阶段5: 测试生成
            logger.info("Running test generation...")
            test_start = time.time()
            test_output = self.generate("测试")
            logger.info(
                f"Test generation completed in {time.time() - test_start:.2f} seconds"
            )
            logger.info(f"Output length: {len(test_output)} characters")

            logger.info("\n" + "=" * 80)
            logger.info("Model initialization completed successfully")
            logger.info("=" * 80)
            return True

        except Exception as e:
            logger.error("\n" + "!" * 80)
            logger.error("MODEL LOADING FAILED")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error("!" * 80)

            # 记录可能的内存错误
            if "CUDA out of memory" in str(e):
                logger.error("CUDA内存不足！建议：")
                logger.error("1. 在Kaggle设置中启用GPU")
                logger.error("2. 减少max_new_tokens参数")
                logger.error("3. 使用更小的模型")

            raise

    def generate(self, keyword):
        """生成故事"""
        try:
            logger.info(f"\nStarting generation for: {keyword}")
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

            # Generation
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
            logger.info(
                f"Generation speed: {outputs.shape[1] / (time.time() - gen_start):.1f} tokens/s"
            )

            # Decoding
            story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = story.split("】的童话故事，")[-1].strip()
            logger.info(f"Final output length: {len(result)} characters")

            return result

        except Exception as e:
            logger.error("\nGeneration failed with error:")
            logger.error(f"Type: {type(e).__name__}")
            logger.error(f"Details: {str(e)}")
            raise

