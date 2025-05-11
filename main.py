import os
import logging
from story_generator import (
    ensure_model_downloaded as ensure_story_model,
    generate_story,
)
from image_generator import (
    ensure_model_downloaded as ensure_image_model,
    generate_image,
)
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    keyword = input("请输入关键字（例如：勇敢的小兔子）：").strip()
    if not keyword:
        print("请输入有效关键词")
        return

    # 模型路径 - 修正拼写错误
    story_model_name = "uer/gpt2-chinese-cluecorpussmall"
    story_model_path = os.path.join("models", story_model_name.replace("/", "_"))
    image_model_id = "CompVis/stable-diffusion-v1-4"
    image_model_path = os.path.join("models", image_model_id.replace("/", "_"))

    # 确保模型已下载
    if not ensure_story_model(story_model_name, story_model_path):
        print("故事模型加载失败，无法继续。")
        return
    if not ensure_image_model(image_model_id, image_model_path):
        print("图片模型加载失败，无法继续。")
        return

    try:
        # 生成故事
        story = generate_story(keyword, story_model_path)
        print("\n生成的故事:\n")
        print(story)

        # 保存故事
        os.makedirs("output", exist_ok=True)
        story_path = os.path.join("output", "story.txt")
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(story)
        print(f"\n故事已保存到 {story_path}")

        # 生成并保存图片
        image = generate_image(story, image_model_path)
        image_path = os.path.join("output", "illustration.png")
        image.save(image_path)
        print(f"插图已保存到 {image_path}")

        # 尝试打开图片
        try:
            image.show()
        except:
            print("无法自动显示图片，请手动打开查看")

    except Exception as e:
        logger.error(f"生成过程中出错: {e}")


if __name__ == "__main__":
    main()

