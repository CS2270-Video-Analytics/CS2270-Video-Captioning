from config import Config
from ..model import Model
import torch
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import os 
from typing import Optional, Dict
from torchvision import transforms
from time import time
from torchvision import transforms

class GPTModel():

    def __init__(self, model_name="gpt-4o-mini"):
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        OpenAI.api_key = os.environ['OPENAI_API_KEY']  # Set API key from environment
        self.model_client = OpenAI()
        self.model_name = model_name


    def preprocess_data(self, data_stream:"Tensor", text_input:Optional[str]):
        pass
    
    def run_inference(self, query: str, **kwargs):
        info = {}
        messages = [{"role": "user", "content": query}]

        response = self.model_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip(), info

