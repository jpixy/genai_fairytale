#!/usr/bin/env python3
from model_loader import StoryGenerator
from config import logger, MODEL_CONFIG
import time
import sys


def display_help():
    print(f"""
    {"*" * 40}
    童话生成器 {MODEL_CONFIG["name"]}
    {"*" * 40}
    使用说明：
    1. 首次使用会自动下载模型(约3GB)
    2. 输入中文主角名称(如: 勇敢的小刺猬)
    3. 输入 q 退出程序
    
    常见问题处理：
    - 卡在下载: 检查网络或使用代理
    - 加载失败: 运行 rm -rf models/Qwen2-1.5B
    - 生成错误: 更换主角名称重试
    {"*" * 40}
    """)


def main():
    display_help()
    generator = StoryGenerator()

    # 模型加载
    print("\n🛠️ 正在初始化模型...")
    try:
        start_time = time.time()
        if not generator.load_model():
            print("❌ 模型初始化失败，请查看日志")
            sys.exit(1)
        print(f"🕒 初始化耗时: {time.time() - start_time:.1f}秒")
    except Exception as e:
        print(f"\n💢 严重错误: {str(e)}")
        sys.exit(1)

    # 交互循环
    while True:
        try:
            keyword = input("\n🖋️ 请输入故事主角 (q退出): ").strip()
            if keyword.lower() in ["q", "quit"]:
                break

            if not keyword:
                print("⚠️ 请输入有效名称")
                continue

            print(f"\n✨ 正在为【{keyword}】创作童话...")
            try:
                start_gen = time.time()
                story = generator.generate(keyword)
                print(f"\n📜 生成结果 (耗时: {time.time() - start_gen:.1f}s)")
                print("-" * 60)
                print(story)
                print("-" * 60)
            except Exception as e:
                print(f"❌ 生成失败: {str(e)}")

        except KeyboardInterrupt:
            print("\n⏹️ 中断操作，输入q退出")
            continue

    print("\n🎉 感谢使用童话生成器！")


if __name__ == "__main__":
    main()

