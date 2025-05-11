from flask import Flask, render_template, request, jsonify, send_from_directory
from openai_helper import generate_story, generate_image
import os
from config import Config
import logging

app = Flask(__name__)
app.config.from_object(Config)
logging.basicConfig(level=logging.INFO)


@app.route("/", methods=["GET", "POST"])
def index():
    story = ""
    image_file = None

    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        if keyword:
            try:
                story = generate_story(keyword)
                if story:
                    image_file = generate_image(f"童话故事插图：{story[:100]}...")
            except Exception as e:
                logging.error(f"生成失败: {e}")
                story = f"生成过程中出错: {str(e)}"

    return render_template("index.html", story=story, image_file=image_file)


@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

