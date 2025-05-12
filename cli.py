from hf_client import HFClient
import os
from datetime import datetime
from config import logger


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_story(text, keyword):
    ensure_dir("outputs/stories")
    filename = f"{keyword}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    path = os.path.join("outputs/stories", filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"故事保存到: {path}")
    return path


def save_image(image, keyword):
    ensure_dir("outputs/images")
    filename = f"{keyword}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    path = os.path.join("outputs/images", filename)

    image.save(path)
    logger.info(f"插图保存到: {path}")
    return path


def generate_fairytale(keyword):
    client = HFClient()
    results = {}

    try:
        logger.info(f"开始生成童话: {keyword}")

        # 生成故事
        logger.info("步骤1/2: 生成故事文本")
        story = client.generate_story(keyword)
        story_path = save_story(story, keyword)
        results["story"] = story
        results["story_path"] = story_path

        # 生成插图
        logger.info("步骤2/2: 生成故事插图")
        image = client.generate_image(story[:100])
        image_path = save_image(image, keyword)
        results["image_path"] = image_path

        logger.info("童话生成完成!")
        return results

    except Exception as e:
        logger.error(f"生成过程中出错: {e}")
        raise


if __name__ == "__main__":
    print("=== 童话故事生成器 (命令行版) ===")
    keyword = input("请输入故事主角（例如：勇敢的小熊）：").strip()

    try:
        results = generate_fairytale(keyword)
        print(
            f"\n生成完成！\n故事文件: {results['story_path']}\n插图文件: {results['image_path']}"
        )

    except Exception as e:
        print(f"\n生成失败: {e}")

