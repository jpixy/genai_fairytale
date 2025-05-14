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
from config import logger
from pathlib import Path
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.static_folder = "static"
generator = StoryGenerator()

output_dir = Path("outputs/stories")
output_dir.mkdir(parents=True, exist_ok=True)


@app.before_first_request
def initialize():
    try:
        logger.info("Initializing model...")
        generator.load_model()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        if not keyword:
            flash("请输入故事主角名称", "warning")
            return redirect(url_for("index"))

        try:
            story = generator.generate(keyword)

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
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            flash(f"生成失败: {str(e)}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/download/<filename>")
def download(filename):
    try:
        return send_from_directory(output_dir, filename, as_attachment=True)
    except FileNotFoundError:
        flash("文件不存在", "warning")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

