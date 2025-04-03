from config import Config
from ..model import VisionLanguageModel
import torch
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import os 
from typing import Optional, Dict
from torchvision import transforms
from time import time
from torchvision import transforms

class OpenAI(VisionLanguageModel):

    def __init__(self, model_name:str="gpt-4o-mini", model_precision = torch.float16, system_eval:bool = False):
        super().__init__()
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        OpenAI.api_key = os.environ['OPENAI_API_KEY']  # Set API key from environment
        self.model_client = OpenAI()
        self.model_name = model_name

        #auxilliary attributes for Tensor to image conversion
        self.system_eval = system_eval

    def preprocess_data(self, data_stream: torch.Tensor, prompt:Optional[str]=None):
        
        base64_encoded_images = self.pil_to_base64(self.to_pil_image(data_stream))


        return base64_encoded_images
    
    
    def run_inference(self, data_stream: torch.Tensor, **kwargs):

        if self.system_eval:
            start_time = time.now()

        processed_inputs = self.preprocess_data(data_stream)

        info = {}

        messages = [
                    {"role": "system", "content": kwargs['system_content']},
                    {"role": "user", "content": [
                        {"type": "text", "text": kwargs['prompt']},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{preprocess_data}"}}
                    ]}]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs['temperature'],
                top_p = kwargs['top_p'],
                max_tokens=kwargs['max_tokens'],
                frequency_penalty=kwargs['frequency_penalty'],
                presence_penalty=kwargs['presence_penalty'],
                stop_token=kwargs['stop_tokens']
            )
            info['error'] = None
        except openai.APIError as e:
            info['error'] = e
        
        if self.system_eval:
            end_time = time.now()
            elapsed = end_time - start_time
            info['time'] = elapsed

        return response.choices[0].message.content.strip(), info




