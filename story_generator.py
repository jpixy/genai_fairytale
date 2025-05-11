from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
from accelerate import init_empty_weights

logger = logging.getLogger(__name__)


def ensure_model_downloaded(model_name, model_path):
    try:
        # 确保模型目录存在
        os.makedirs(model_path, exist_ok=True)

        # 检查是否已经有模型文件
        if os.path.exists(os.path.join(model_path, "config.json")):
            logger.info(f"模型 {model_name} 已存在于 {model_path}")
            return True

        logger.info(f"下载模型 {model_name} 到 {model_path}...")

        # 使用accelerate优化模型加载
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        return True
    except Exception as e:
        logger.error(f"下载模型 {model_name} 失败: {e}")
        return False


def generate_story(keyword, model_path):
    try:
        logger.info(f"正在从 {model_path} 加载故事生成模型...")

        # 检查必要的文件是否存在
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                raise FileNotFoundError(f"在 {model_path} 找不到 {file}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)

        prompt = f"从前，有一个{keyword}，它住在一个美丽的小村庄里。一天，{keyword}决定去探险。"
        logger.info(f"使用提示: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_length=300,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        story = tokenizer.decode(output[0], skip_special_tokens=True)
        return story
    except Exception as e:
        logger.error(f"故事生成失败: {e}")
        raise

