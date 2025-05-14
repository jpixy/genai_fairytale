from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    flash,
    redirect,
    url_for,
)
from model_loader import StoryGenerator
from config import logger, MODEL_CONFIG
from pathlib import Path
import shutil
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 设置Flask密钥
app.static_folder = "static"
generator = StoryGenerator()

# 确保输出目录存在
output_dir = Path("outputs/stories")
output_dir.mkdir(parents=True, exist_ok=True)


@app.before_first_request
def initialize():
    """初始化模型加载"""
    try:
        logger.info("Initializing story generator...")
        if not generator.model_dir.exists():
            flash("首次使用需要下载模型，请稍候...", "info")
            generator.load_model()
            flash("模型加载完成！", "success")
        else:
            generator.load_model()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        flash(f"模型加载失败: {str(e)}", "danger")
        # 清理可能损坏的下载
        if generator.model_dir.exists():
            shutil.rmtree(generator.model_dir)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        if not keyword:
            flash("请输入故事主角名称", "warning")
            return redirect(url_for("index"))

        try:
            logger.info(f"Generating story for: {keyword}")
            story = generator.generate(keyword)

            # 保存故事文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{keyword}_{timestamp}.txt"
            filepath = output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(story)

            return render_template(
                "index.html",
                story=story,
                keyword=keyword,
                filename=filename,
                char_count=len(story),
                generate_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            flash(f"生成失败: {str(e)}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/download/<filename>")
def download(filename):
    """下载生成的故事文件"""
    try:
        return send_from_directory(
            output_dir, filename, as_attachment=True, mimetype="text/plain"
        )
    except FileNotFoundError:
        flash("文件不存在或已被删除", "warning")
        return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(e):
    logger.critical(f"Server error: {e}")
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

