import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from config import logger, MODEL_CONFIG
import os
import json


class StoryGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"设备类型: {self.device.upper()}")
        self.model = None
        self.tokenizer = None
        self.model_dir = Path(MODEL_CONFIG["local_dir"])

    def _patch_model_config(self):
        """强制修改模型配置绕过flash_attn检查"""
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r+", encoding="utf-8") as f:
                config = json.load(f)
                # 关键修改：禁用所有flash attention相关设置
                config["use_flash_attention_2"] = False
                config["_attn_implementation"] = "eager"
                if "attn_config" in config:
                    config["attn_config"]["attn_impl"] = "eager"
                f.seek(0)
                json.dump(config, f, indent=2)
                f.truncate()
                logger.info("模型配置文件修改完成")

    def load_model(self):
        """完全绕过flash_attn的加载方法"""
        try:
            # 准备目录
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # 下载模型（如果不存在）
            if not any(self.model_dir.glob("*")):
                logger.info("正在下载模型...")
                snapshot_download(
                    repo_id=MODEL_CONFIG["name"],
                    local_dir=self.model_dir,
                    ignore_patterns=["*.bin"],
                    resume_download=True,
                )

            # 关键步骤：修改配置文件
            self._patch_model_config()

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True, padding_side="left"
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 强制使用eager attention加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float32,  # 兼容所有设备
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",  # 强制使用普通注意力
            ).eval()

            logger.info("✅ 模型加载成功！")
            return True

        except Exception as e:
            logger.error(f"加载失败: {str(e)}")
            raise RuntimeError(f"""
            模型加载失败，请按顺序尝试：
            1. 删除模型缓存: rm -rf {self.model_dir}
            2. 检查网络连接
            3. 确保磁盘剩余空间 > 10GB
            4. 降低transformers版本: pip install transformers==4.40.1
            """)

    def generate(self, keyword):
        """生成童话故事"""
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

