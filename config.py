import os


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    STORY_MODEL = "gpt-3.5-turbo"
    IMAGE_MODEL = "dall-e-3"
    IMAGE_SIZE = "1024x1024"
    IMAGE_QUALITY = "standard"
    IMAGE_STYLE = "vivid"
