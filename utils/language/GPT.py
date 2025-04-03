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
        messages = [{"role": "system", "content": kwargs['system_content']},{"role": "user", "content": query}]

        try:
            response = self.model_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs['temperature'],
                top_p = kwargs['top_p'],
                max_tokens=kwargs['max_tokens'],
                frequency_penalty=kwargs['frequency_penalty'],
                presence_penalty=kwargs['presence_penalty'],
                stop_token=kwargs['stop_token']
            )

            info['error'] = None
        except openai.APIError as e:
            info['error'] = e

        return response.choices[0].message.content.strip(), info

