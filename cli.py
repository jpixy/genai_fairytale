import argparse
from model_loader import QwenStoryGenerator
from config import logger, MODEL_CONFIG
import time


def main():
    parser = argparse.ArgumentParser(description="Qwen 童话生成器")
    parser.add_argument("keyword", type=str, help="童话故事的主角名称")
    args = parser.parse_args()

    print("=== Qwen 童话生成器 ===")
    print(f"Model: {MODEL_CONFIG['name']}")

    generator = QwenStoryGenerator()

    try:
        # 初始化
        print("\n加载模型中...")
        start = time.time()
        generator.load_model()
        print(f"准备就绪 (耗时: {time.time() - start:.1f}s)")

        # 生成
        print("\n✨ 创作中...")
        start_gen = time.time()
        story = generator.generate(args.keyword)

        # 输出
        print(f"\n📖 《{args.keyword}的童话》")
        print("-" * 40)
        print(story)
        print("-" * 40)
        print(f"⏱️ 生成耗时: {time.time() - start_gen:.1f}s")

    except Exception as e:
        logger.error(f"Error: {e}")
        print("生成失败，请查看日志")


if __name__ == "__main__":
    main()

