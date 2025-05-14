#!/usr/bin/env python3
from model_loader import StoryGenerator
from config import logger
import argparse


def main():
    parser = argparse.ArgumentParser(description="童话故事生成器")
    parser.add_argument("keyword", type=str, help="故事主角名称")
    args = parser.parse_args()

    generator = StoryGenerator()

    try:
        generator.load_model()
        story = generator.generate(args.keyword)
        print("\n生成的童话故事:")
        print("=" * 60)
        print(story)
        print("=" * 60)
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

