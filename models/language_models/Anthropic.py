from config import Config
from ..model import Model
import torch
from dotenv import load_dotenv, find_dotenv
from anthropic import Anthropic, APIStatusError, APIConnectionError
import os 
from typing import Optional, Dict
from torchvision import transforms
from time import time
from torchvision import transforms

class Anthropic():

    def __init__(self, model_name="claude-3-5-haiku-latest"):
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        
        self.model_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        self.model_name = model_name


    def preprocess_data(self, data_stream:"Tensor", text_input:Optional[str]):
        pass
    
    def run_inference(self, query: str, **kwargs):
        info = {}
        messages = [{"role": "system", "content": kwargs['system_content']},{"role": "user", "content": query}]

        try:
            response = self.model_client.messages.create(
                model = self.model_name,
                max_tokens = kwargs['max_tokens'],
                temperature = kwargs['temperature'],
                top_p = kwargs['top_p'],
                messages = messages,
            )

            info['error'] = None
        except APIStatusError or APIConnectionError as e:
            info['error'] = e

        return response.content[0].text.strip(), info

