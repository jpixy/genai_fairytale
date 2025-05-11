from flask import Flask, render_template, request, send_from_directory
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

app = Flask(__name__)

# 模型路径
story_model_name = "uer/gpt2-chinese-cluecorpussmall"
story_model_path = os.path.join("models", "gpt2-chinese-cluecorpussmall")
image_model_id = "CompVis/stable-diffusion-v1-4"
image_model_path = os.path.join("models", "stable-diffusion-v1-4")

# 确保模型已下载
ensure_story_model(story_model_name, story_model_path)
ensure_image_model(image_model_id, image_model_path)


@app.route("/", methods=["GET", "POST"])
def index():
    story = ""
    image_path = None
    if request.method == "POST":
        keyword = request.form.get("keyword")
        story = generate_story(keyword, story_model_path)
        image = generate_image(story, image_model_path)

        # 保存故事到文件
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        story_path = os.path.join(output_dir, "story.txt")
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(story)

        # 保存插图到文件
        image_path = os.path.join(output_dir, "illustration.png")
        image.save(image_path)

        # 保存插图到静态目录以便网页展示
        static_image_path = os.path.join("static", "images", "illustration.png")
        image.save(static_image_path)
    return render_template("index.html", story=story, image_path=static_image_path)


@app.route("/static/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(os.path.join("static", "images"), filename)


if __name__ == "__main__":
    app.run(debug=True)
