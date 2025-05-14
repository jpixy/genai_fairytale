import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from config import logger, MODEL_CONFIG
import time
import shutil


class StoryGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.model_dir = Path(MODEL_CONFIG["local_dir"])
        self._log_system_info()

    def _log_system_info(self):
        logger.info("=" * 80)
        logger.info("Initializing Story Generator - System Diagnostics")
        logger.info(f"Device: {self.device.upper()}")
        logger.info(f"Model: {MODEL_CONFIG['name']}")
        logger.info(f"Cache dir: {self.model_dir}")

        total, used, free = shutil.disk_usage("/")
        logger.info(f"Disk - Total: {total // (2**30)}GB | Free: {free // (2**30)}GB")

        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.memory_allocated() // 1024**2}MB/{torch.cuda.max_memory_allocated() // 1024**2}MB"
            )
        logger.info("=" * 80)

    def _download_with_retry(self):
        for attempt in range(1, MODEL_CONFIG["max_retries"] + 1):
            try:
                logger.info(f"Download Attempt {attempt}/{MODEL_CONFIG['max_retries']}")
                start_time = time.time()

                snapshot_download(
                    repo_id=MODEL_CONFIG["name"],
                    local_dir=self.model_dir,
                    resume_download=True,
                    local_dir_use_symlinks=False,
                    token=None,
                    max_workers=4,
                )

                logger.info(
                    f"Download completed in {time.time() - start_time:.1f} seconds"
                )
                return True

            except Exception as e:
                wait_time = MODEL_CONFIG["retry_delay"] * attempt
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        return False

    def _verify_model_files(self):
        required_files = [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "generation_config.json",
        ]

        logger.info("Verifying downloaded files...")
        all_valid = True

        for file in required_files:
            filepath = self.model_dir / file
            if filepath.exists():
                size = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"Found: {file.ljust(25)} {size:.1f}MB")
            else:
                logger.error(f"Missing: {file}")
                all_valid = False

        return all_valid

    def load_model(self):
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model directory: {self.model_dir}")
            logger.info(f"Contents: {list(self.model_dir.glob('*'))}")

            if not any(self.model_dir.glob("*.safetensors*")):
                logger.info("No model files found, starting download...")
                if not self._download_with_retry():
                    raise RuntimeError("Model download failed after retries")

                if not self._verify_model_files():
                    raise RuntimeError("Model files verification failed")
            else:
                logger.info("Found existing model files")
                if not self._verify_model_files():
                    logger.warning("Existing files are incomplete, re-downloading...")
                    shutil.rmtree(self.model_dir)
                    return self.load_model()

            logger.info("Loading tokenizer...")
            start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer loaded in {time.time() - start_time:.1f}s")

            logger.info("Loading model...")
            start_time = time.time()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).eval()
            logger.info(f"Model loaded in {time.time() - start_time:.1f}s")

            logger.info("Running test generation...")
            test_start = time.time()
            test_output = self.generate("测试")
            logger.info(f"Test passed in {time.time() - test_start:.1f}s")
            logger.info(f"Sample output: {test_output[:100]}...")

            logger.info("=" * 80)
            logger.info("Model initialization completed successfully")
            logger.info("=" * 80)
            return True

        except Exception as e:
            logger.error("=" * 80)
            logger.error("MODEL LOADING FAILED")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("=" * 80)
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
            logger.info(f"Generation completed in {time.time() - gen_start:.1f}s")
            logger.info(
                f"Speed: {outputs.shape[1] / (time.time() - gen_start):.1f} tokens/s"
            )

            story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = story.split("】的童话故事，")[-1].strip()
            logger.info(f"Final length: {len(result)} characters")

            return result

        except Exception as e:
            logger.error("Generation failed")
            logger.error(f"Error: {str(e)}")
            raise

