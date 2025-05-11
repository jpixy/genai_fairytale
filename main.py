import os
from openai_helper import generate_story, generate_image


def test_story_generation(keyword="勇敢的小兔子"):
    """测试故事生成功能"""
    try:
        print(f"正在生成关于'{keyword}'的故事...")
        story = generate_story(keyword)

        # 保存故事
        os.makedirs("output", exist_ok=True)
        story_path = os.path.join("output", "story.txt")
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(story)

        print(f"✔ 故事生成成功，已保存到 {story_path}")
        print("故事开头：", story[:100] + "...")
        return story
    except Exception as e:
        print(f"✖ 故事生成失败: {e}")
        return None


def test_image_generation(prompt="一只勇敢的小兔子在森林里探险"):
    """测试图片生成功能"""
    try:
        print(f"正在生成图片：'{prompt}'...")
        image_file = generate_image(prompt, save_path="output")

        image_path = os.path.join("output", image_file)
        print(f"✔ 图片生成成功，已保存到 {image_path}")
        return image_path
    except Exception as e:
        print(f"✖ 图片生成失败: {e}")
        return None


def cleanup():
    """清理测试文件"""
    test_files = [
        os.path.join("output", "story.txt"),
        os.path.join("output", "illustration_*.png"),
    ]

    for file_pattern in test_files:
        for file in glob.glob(file_pattern):
            try:
                os.remove(file)
                print(f"清理文件: {file}")
            except:
                pass


if __name__ == "__main__":
    import glob

    print("=" * 50)
    print("开始测试童话生成器核心功能")
    print("=" * 50)

    # 清理旧文件
    print("\n[准备阶段]")
    cleanup()

    # 测试故事生成
    print("\n[测试1/2] 故事生成功能")
    story = test_story_generation()

    # 测试图片生成
    if story:
        print("\n[测试2/2] 插图生成功能")
        test_image_generation(story[:100])  # 使用故事开头作为提示词

    # 验证结果
    print("\n[验证结果]")
    story_exists = os.path.exists(os.path.join("output", "story.txt"))
    image_exists = len(glob.glob(os.path.join("output", "illustration_*.png"))) > 0

    if story_exists and image_exists:
        print("✔ 所有功能测试通过！")
        print("生成的文件保存在 output/ 目录")
    else:
        print("✖ 测试失败，请检查：")
        if not story_exists:
            print("- 故事生成未完成")
        if not image_exists:
            print("- 图片生成未完成")

    print("\n测试完成")

