import os 
from config import HF_API_TOKEN, MODEL_NAME, EMBED_MODEL_ID, IMAGE_MODEL_ID
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer
import torch

def test_env_loading():
    assert HF_API_TOKEN is not None
    assert MODEL_NAME is not None
    assert EMBED_MODEL_ID is not None
    assert IMAGE_MODEL_ID is not None

def test_text_model_loading():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_API_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_API_TOKEN , device_map= "cpu")
    inputs = tokenizer("hello nutriguard!", return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=5)
    assert out is not None

def test_embedding_model_loading():
    embed_model = SentenceTransformer(EMBED_MODEL_ID)
    vec = embed_model.encode("hello nutriguard!")
    assert vec is not None
    assert len(vec) > 0

def test_image_model_loading():
    processor = AutoProcessor.from_pretrained(IMAGE_MODEL_ID, use_auth_token=HF_API_TOKEN)
    model = AutoModel.from_pretrained(IMAGE_MODEL_ID , use_auth_token=HF_API_TOKEN , device_map= "cpu")
    assert processor is not None
    assert model is not None


    