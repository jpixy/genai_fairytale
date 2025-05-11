from openai import OpenAI
from config import Config
import logging
import base64
import os

client = OpenAI(api_key=Config.OPENAI_API_KEY)
logger = logging.getLogger(__name__)


def generate_story(keyword):
    try:
        response = client.chat.completions.create(
            model=Config.STORY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位专业的儿童文学作家，擅长创作富有教育意义的童话故事",
                },
                {
                    "role": "user",
                    "content": f"""请创作一个关于{keyword}的童话故事，要求：
1. 包含完整的故事结构：开头-发展-高潮-结局
2. 语言生动有趣，适合6-12岁儿童
3. 300字左右
4. 结尾包含故事寓意""",
                },
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"故事生成失败: {e}")
        raise


def generate_image(prompt, save_path="static/images"):
    os.makedirs(save_path, exist_ok=True)
    try:
        response = client.images.generate(
            model=Config.IMAGE_MODEL,
            prompt=f"儿童童话风格插图，内容：{prompt}。要求：色彩鲜艳，卡通风格，适合儿童观看",
            size=Config.IMAGE_SIZE,
            quality=Config.IMAGE_QUALITY,
            style=Config.IMAGE_STYLE,
            n=1,
            response_format="b64_json",
        )

        image_data = base64.b64decode(response.data[0].b64_json)
        filename = f"illustration_{hash(prompt)}.png"
        filepath = os.path.join(save_path, filename)

        with open(filepath, "wb") as f:
            f.write(image_data)

        return filename
    except Exception as e:
        logger.error(f"图片生成失败: {e}")
        raise
