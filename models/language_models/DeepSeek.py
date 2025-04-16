from config import Config
from ..model import LanguageModel
import torch
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import openai
import os 
from typing import Optional, Dict
from torchvision import transforms
from time import time
from torchvision import transforms

class DeepSeek(LanguageModel):
    def __init__(self, model_params:dict, model_name:str="deepseek-chat", model_precision = torch.float16, system_eval:bool = False):
        super().__init__(model_params=model_params, model_precision=model_precision, system_eval=system_eval)
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        OpenAI.api_key = os.environ['DEEPSEEK_API_KEY']  # Set API key from environment
        self.model_client = OpenAI()
        self.model_name = model_name


    def preprocess_data(self, data_stream:str, text_input:Optional[str]):
        pass
    
    def run_inference(self, data_stream: str, **kwargs):
        info = {}
        messages = [{"role": "system", "content": kwargs['system_content']},{"role": "user", "content": query}]

        if self.system_eval:
            start_time = time.now()

        try:
            request_params = dict(model=self.model_name,
                messages=messages,
                temperature=self.model_params['temperature'],
                top_p = self.model_params['top_p'],
                max_tokens=self.model_params['max_tokens'],
                frequency_penalty=self.model_params['frequency_penalty'],
                presence_penalty=self.model_params['presence_penalty'])

            request_params = {k: v for k, v in request_params.items() if v is not None}
            stop_tokens = self.model_params.get("stop_tokens")
            if isinstance(stop_tokens, list) and stop_tokens:
                request_params["stop"] = stop_tokens
            
            response = self.model_client.chat.completions.create(**request_params)

            info['error'] = None
        except openai.APIError as e:
            info['error'] = e
        
        if self.system_eval:
            end_time = time.now()
            elapsed = end_time - start_time
            info['time'] = elapsed


        return response.choices[0].message.content.strip(), info

