from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def ensure_model_downloaded(model_name, model_path):
    if not os.path.exists(model_path):
        print(f"模型 {model_name} 未找到，正在下载...")
        AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path)
        AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
        print(f"模型已下载到 {model_path}")


def generate_story(keyword, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    prompt = (
        f"从前，有一个{keyword}，它住在一个美丽的小村庄里。一天，{keyword}决定去探险。"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=300, no_repeat_ngram_size=2)
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story
