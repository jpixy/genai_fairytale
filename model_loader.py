import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from config import logger, MODEL_CONFIG


class QwenStoryGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")
        self.model = None
        self.tokenizer = None
        self.model_dir = Path(MODEL_CONFIG["local_dir"])

    def _download_model(self):
        """下载模型（如果不存在）"""
        if not any(self.model_dir.glob("*.safetensors")):
            logger.info("Downloading model...")
            snapshot_download(
                repo_id=MODEL_CONFIG["name"],
                local_dir=self.model_dir,
                ignore_patterns=["*.bin"],
                resume_download=False,
            )

    def load_model(self):
        """加载 Qwen 指令模型"""
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self._download_model()

            # 加载带有 Qwen 专用模板的分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True
            )

            # 4位量化加载以节省显存
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).eval()

            logger.info("Model ready")

        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise

    def generate(self, keyword):
        """生成结构化童话"""
        messages = [
            {"role": "system", "content": MODEL_CONFIG["system_prompt"]},
            {"role": "user", "content": f"请以【{keyword}】为主角创作童话"},
        ]

        # 应用 Qwen 专用对话模板
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            self.device
        )

        # 确保 attention_mask 设置正确
        attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(self.device)

        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            **MODEL_CONFIG["generation"],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._format_story(full_text)

    def _format_story(self, text):
        """后处理增强可读性"""
        # 移除模板标签
        for tag in ["<|im_start|>", "<|im_end|>"]:
            text = text.replace(tag, "")

        # 确保结构完整
        if "[背景]" not in text:
            text = "[背景]\n" + text
        if "[寓意]" not in text:
            text += "\n[寓意] 坚持梦想终会成功"

        return text.strip()

