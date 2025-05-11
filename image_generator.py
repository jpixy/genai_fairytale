from diffusers import StableDiffusionPipeline
import torch
import os
import logging
from accelerate import init_empty_weights

logger = logging.getLogger(__name__)


def ensure_model_downloaded(model_id, model_path):
    try:
        os.makedirs(model_path, exist_ok=True)

        if os.path.exists(os.path.join(model_path, "model_index.json")):
            logger.info(f"模型 {model_id} 已存在于 {model_path}")
            return True

        logger.info(f"下载模型 {model_id} 到 {model_path}...")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        return True
    except Exception as e:
        logger.error(f"下载模型 {model_id} 失败: {e}")
        return False


def generate_image(prompt, model_path):
    try:
        logger.info("正在加载图片生成模型...")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            low_cpu_mem_usage=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")
        pipe = pipe.to(device)

        logger.info(f"正在生成图片，提示词: {prompt}")
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

        return image
    except Exception as e:
        logger.error(f"图片生成失败: {e}")
        raise

