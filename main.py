import os
from story_generator import (
    ensure_model_downloaded as ensure_story_model,
    generate_story,
)
from image_generator import (
    ensure_model_downloaded as ensure_image_model,
    generate_image,
)
from PIL import Image


def main():
    keyword = input("请输入关键字（例如：勇敢的小兔子）：")

    # 模型路径
    story_model_name = "uer/gpt2-chinese-cluecorpussmall"
    story_model_path = os.path.join("models", "gpt2-chinese-cluecorpussmall")
    image_model_id = "CompVis/stable-diffusion-v1-4"
    image_model_path = os.path.join("models", "stable-diffusion-v1-4")

    # 确保模型已下载
    ensure_story_model(story_model_name, story_model_path)
    ensure_image_model(image_model_id, image_model_path)

    # 生成故事
    story = generate_story(keyword, story_model_path)

    # 保存故事到文件
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    story_path = os.path.join(output_dir, "story.txt")
    with open(story_path, "w", encoding="utf-8") as f:
        f.write(story)
    print(f"故事已保存到 {story_path}")

    # 生成插图
    image = generate_image(story, image_model_path)
    image_path = os.path.join(output_dir, "illustration.png")
    image.save(image_path)
    print(f"插图已保存到 {image_path}")


if __name__ == "__main__":
    main()

