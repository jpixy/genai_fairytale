import requests
from PIL import Image
import io
from config import (
    logger,
    HF_API_TOKEN,
    TEXT_MODEL,
    IMAGE_MODEL,
    STORY_SETTINGS,
    IMAGE_SETTINGS,
)


class HFClient:
    def __init__(self):
        if not HF_API_TOKEN:
            raise ValueError("Hugging Face API Token 未配置！请检查config.py")

        self.headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        logger.info("HuggingFace API客户端初始化完成")

    def generate_story(self, keyword):
        """生成童话故事"""
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("关键词必须是非空字符串")

        logger.info(f"开始生成故事，关键词: {keyword}")
        prompt = f"创作一个关于{keyword}的儿童童话故事，300字左右，包含教育意义"

        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{TEXT_MODEL}",
                headers=self.headers,
                json={
                    "inputs": prompt,
                    "parameters": STORY_SETTINGS,
                    "options": {"wait_for_model": True},
                },
                timeout=60,
            )

            response.raise_for_status()  # 自动处理4xx/5xx错误

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    if generated_text:
                        logger.info("故事生成成功")
                        return generated_text

            raise Exception("API返回了无效的响应格式")

        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {str(e)}")
            raise Exception(f"故事生成请求失败: {str(e)}")
        except Exception as e:
            logger.error(f"处理响应时出错: {str(e)}")
            raise Exception(f"故事生成失败: {str(e)}")

    def generate_image(self, prompt):
        """生成童话插图"""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("提示词必须是非空字符串")

        logger.info(f"开始生成插图，提示词: {prompt[:50]}...")

        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}",
                headers=self.headers,
                json={
                    "inputs": f"儿童绘本风格，{prompt}",
                    "parameters": IMAGE_SETTINGS,
                },
                timeout=120,
            )

            response.raise_for_status()

            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                if image:
                    logger.info("插图生成成功")
                    return image

            raise Exception("API返回了无效的图片数据")

        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {str(e)}")
            raise Exception(f"插图生成请求失败: {str(e)}")
        except Exception as e:
            logger.error(f"处理图片时出错: {str(e)}")
            raise Exception(f"插图生成失败: {str(e)}")

