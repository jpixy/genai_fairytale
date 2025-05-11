setup:
	pip install -r requirements.txt
	export HF_ENDPOINT=https://hf-mirror.com

run:
	python main.py

run_web:
	python app.py

