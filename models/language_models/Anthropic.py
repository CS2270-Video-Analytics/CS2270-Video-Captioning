from config import Config
from ..model import LanguageModel
import torch
from dotenv import load_dotenv, find_dotenv
from anthropic import Anthropic, APIStatusError, APIConnectionError
import os 
from typing import Optional, Dict
from torchvision import transforms
from time import time

class AnthropicText(LanguageModel):
    def __init__(self, model_params: Dict, model_name: str = "claude-3-5-haiku-latest", model_precision = torch.float16, system_eval: bool = False):
        super().__init__(model_params=model_params, model_precision=model_precision, system_eval=system_eval)
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        self.model_client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        self.model_name = model_name

    def preprocess_data(self, data_stream: str, text_input: Optional[str]):
        pass
    
    def run_inference(self, data_stream: str, **kwargs):
        info = {}
        messages = [
            {"role": "system", "content": kwargs['system_content']},
            {"role": "user", "content": data_stream}
        ]

        if self.system_eval:
            start_time = time.time()

        try:
            response = self.model_client.messages.create(
                model=self.model_name,
                messages=messages,
                temperature=self.model_params['temperature'],
                top_p=self.model_params['top_p'],
                max_tokens=self.model_params['max_tokens'],
                frequency_penalty=self.model_params['frequency_penalty'],
                presence_penalty=self.model_params['presence_penalty']
            )
            info['error'] = None
        except (APIStatusError, APIConnectionError) as e:
            info['error'] = e
        
        if self.system_eval:
            end_time = time.time()
            elapsed = end_time - start_time
            info['time'] = elapsed

        return response.content[0].text.strip(), info

