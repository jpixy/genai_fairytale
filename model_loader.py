import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from config import logger, MODEL_CONFIG
import json
from tqdm import tqdm


class StoryGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.tokenizer = None
        self.model_dir = Path(MODEL_CONFIG["local_dir"])

    def _download_with_progress(self):
        snapshot_download(
            repo_id=MODEL_CONFIG["name"],
            local_dir=self.model_dir,
            ignore_patterns=["*.bin"],
            resume_download=True,
            tqdm_class=tqdm,
        )

    def load_model(self):
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)

            if not any(self.model_dir.glob("*")):
                logger.info("Downloading model files...")
                self._download_with_progress()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def generate(self, keyword):
        prompt = f"请用700字创作关于【{keyword}】的童话故事，包含完整的故事结构，语言生动有趣。"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True, padding=True
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

