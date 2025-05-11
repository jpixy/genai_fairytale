import requests
from PIL import Image
import io
import time
from config import *


class HFClient:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    def generate_story(self, keyword):
        """生成童话故事"""
        prompt = f"创作一个关于{keyword}的儿童童话故事，300字左右，包含教育意义"

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{TEXT_MODEL}",
            headers=self.headers,
            json={
                "inputs": prompt,
                "parameters": STORY_SETTINGS,
                "options": {"wait_for_model": True},
            },
        )

        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            raise Exception(f"故事生成失败: {response.text}")

    def generate_image(self, prompt):
        """生成童话插图"""
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}",
            headers=self.headers,
            json={"inputs": f"儿童绘本风格，{prompt}", "parameters": IMAGE_SETTINGS},
        )

        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            raise Exception(f"插图生成失败: {response.text}")
