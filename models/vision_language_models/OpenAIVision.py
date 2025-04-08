from models.model import VisionLanguageModel
import torch
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import openai
import os 
from typing import Optional, Dict
from torchvision import transforms
from time import time
from torchvision import transforms
import pdb

class OpenAIVision(VisionLanguageModel):

    def __init__(self, model_params:dict, model_name:str="gpt-4o-mini", model_precision = torch.float16, system_eval:bool = False):
        super().__init__(model_name = model_name, model_params = model_params, model_precision = model_precision, system_eval = system_eval)
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        OpenAI.api_key = os.environ['OPENAI_API_KEY']  # Set API key from environment
        self.model_client = OpenAI()
        self.model_name = model_name

    def preprocess_data(self, data_stream: torch.Tensor, prompt:Optional[str]=None):
        # Convert tensor to PIL image
        pil_image = self.to_pil_image(data_stream)
        # Convert PIL image to base64
        base64_encoded_images = self.pil_to_base64(pil_image)
        return base64_encoded_images
    
    
    def run_inference(self, data_stream: torch.Tensor, **kwargs):

        if self.system_eval:
            start_time = time.time()
            start_time = time.time()

        processed_inputs = self.preprocess_data(data_stream)
        info = {}

        messages = []
        if 'system_content' in kwargs:
            messages.append({"role": "system", "content": kwargs['system_content']})
        
        messages.append({"role": "user", "content": [
                        {"type": "text", "text": kwargs['prompt']},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{processed_inputs}", "detail": kwargs['detail']}}
                    ]})

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
        
        if self.system_eval:
            end_time = time.time()
            end_time = time.time()
            elapsed = end_time - start_time
            info['time'] = elapsed

        return response.choices[0].message.content.strip(), info




