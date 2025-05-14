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
app.secret_key = os.urandom(24)
app.static_folder = "static"
generator = StoryGenerator()

output_dir = Path("outputs/stories")
output_dir.mkdir(parents=True, exist_ok=True)


@app.before_first_request
def initialize():
    try:
        logger.info("Initializing story generator")
        if not generator.model_dir.exists():
            flash("First use requires model download, please wait", "info")
            generator.load_model()
            flash("Model loaded successfully", "success")
        else:
            generator.load_model()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        flash(f"Model loading failed: {str(e)}", "danger")
        if generator.model_dir.exists():
            shutil.rmtree(generator.model_dir)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        if not keyword:
            flash("Please enter story character name", "warning")
            return redirect(url_for("index"))

        try:
            logger.info(f"Generating story for: {keyword}")
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
                generate_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            flash(f"Generation failed: {str(e)}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/download/<filename>")
def download(filename):
    try:
        return send_from_directory(
            output_dir, filename, as_attachment=True, mimetype="text/plain"
        )
    except FileNotFoundError:
        flash("File does not exist or has been deleted", "warning")
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

