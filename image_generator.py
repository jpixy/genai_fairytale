from diffusers import StableDiffusionPipeline
import torch
import os


def ensure_model_downloaded(model_id, model_path):
    if not os.path.exists(model_path):
        print(f"模型 {model_id} 未找到，正在下载...")
        StableDiffusionPipeline.from_pretrained(model_id, cache_dir=model_path)
        print(f"模型已下载到 {model_path}")


def generate_image(prompt, model_path):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    )

    # 检测是否支持 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    translated_prompt = translate_to_english(prompt)
    image = pipe(translated_prompt).images[0]
    return image


def translate_to_english(text):
    translations = {
        "勇敢的小兔子": "a brave little rabbit",
        "美丽的小村庄": "a beautiful little village",
        "冒险": "adventure",
    }
    for key, value in translations.items():
        text = text.replace(key, value)
    return text

