from flask import Flask, render_template, request, jsonify, send_from_directory
from hf_client import HFClient
from cli import generate_fairytale, save_story, save_image
from datetime import datetime
import os
import logging
from config import logger

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        if not keyword:
            return render_template("index.html", error="请输入有效关键词")

        try:
            logger.info(f"Web请求 - 开始生成: {keyword}")
            results = generate_fairytale(keyword)

            # 准备Web显示
            story_short = (
                results["story"][:150] + "..."
                if len(results["story"]) > 150
                else results["story"]
            )
            image_filename = os.path.basename(results["image_path"])

            logger.info(f"Web请求 - 生成完成: {keyword}")
            return render_template(
                "index.html",
                story=results["story"],
                story_short=story_short,
                image_file=image_filename,
                keyword=keyword,
            )

        except Exception as e:
            logger.error(f"Web生成失败: {str(e)}")
            return render_template("index.html", error=str(e))

    return render_template("index.html")


@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory("outputs/images", filename)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    keyword = data.get("keyword", "").strip()

    if not keyword:
        return jsonify({"error": "Missing keyword"}), 400

    try:
        results = generate_fairytale(keyword)
        return jsonify(
            {
                "story": results["story"],
                "story_path": results["story_path"],
                "image_path": results["image_path"],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
