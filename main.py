from hf_client import HFClient
import os
from datetime import datetime


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def save_story(text, keyword):
    """保存故事到文件"""
    ensure_dir("outputs/stories")
    filename = f"{keyword}_{datetime.now().strftime('%Y%m%d%H%M')}.txt"
    path = os.path.join("outputs/stories", filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return path


def save_image(image, keyword):
    """保存插画到文件"""
    ensure_dir("outputs/images")
    filename = f"{keyword}_{datetime.now().strftime('%Y%m%d%H%M')}.png"
    path = os.path.join("outputs/images", filename)

    image.save(path)
    return path


def main():
    client = HFClient()

    print("=== 童话故事生成器 ===")
    keyword = input("请输入故事主角（例如：勇敢的小熊）：").strip()

    try:
        # 生成故事
        print("\n正在创作故事...")
        story = client.generate_story(keyword)
        story_path = save_story(story, keyword)
        print(f"✓ 故事已保存到 {story_path}")
        print("\n" + story[:200] + "...\n")

        # 生成插图
        print("正在绘制插图...")
        image = client.generate_image(story[:100])  # 使用故事开头作为提示
        image_path = save_image(image, keyword)
        print(f"✓ 插图已保存到 {image_path}")

    except Exception as e:
        print(f"\n生成失败: {e}")


if __name__ == "__main__":
    main()
