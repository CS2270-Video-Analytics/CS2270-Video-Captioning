from config import Config
from ..model import LanguageModel
import torch
from dotenv import load_dotenv, find_dotenv
import openai
from openai import OpenAI
import os 
from typing import Optional, Dict
from torchvision import transforms
from time import time


class OpenAIText(LanguageModel):
    def __init__(self, model_params: Dict, model_name:str="gpt-4o-mini", model_precision = torch.float16, system_eval:bool = False):
        super().__init__(model_params=model_params, model_precision=model_precision, system_eval=system_eval)
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        OpenAI.api_key = os.environ['OPENAI_API_KEY']  # Set API key from environment
        self.model_client = OpenAI()
        self.model_name = model_name

    def preprocess_data(self, data_stream: str, text_input:Optional[str]):
        pass
    
    def run_inference(self, data_stream: str, **kwargs):
        info = {}

        if 'system_content' in kwargs:
            messages.append({"role": "system", "content": kwargs['system_content']})
        messages = [
            {"role": "user", "content": data_stream}
        ]

        try:
            request_params = dict(model=self.model_name,
                messages=messages,
                temperature=self.model_params['temperature'],
                top_p = self.model_params['top_p'],
                max_tokens=self.model_params['max_tokens'],
                frequency_penalty=self.model_params['frequency_penalty'],
                presence_penalty=self.model_params['presence_penalty'])
            if self.model_params['stop_tokens'] and type(self.model_params['stop_tokens']) is list:
                request_params['stop'] = self.model_params['stop_tokens']
            
            response = self.model_client.chat.completions.create(**request_params)

            info['error'] = None
        except openai.APIError as e:
            info['error'] = e

        return response.choices[0].message.content.strip(), info

