from dotenv import load_dotenv
import os

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

MODEL_NAME = os.getenv("MODEL_NAME")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID")