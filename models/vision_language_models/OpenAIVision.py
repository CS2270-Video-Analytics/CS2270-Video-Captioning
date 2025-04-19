from models.model import VisionLanguageModel
import torch
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
import openai
import os 
from typing import Optional, Dict
from torchvision import transforms
import pdb
import asyncio

class OpenAIVision(VisionLanguageModel):

    def __init__(self, model_params:dict, model_name:str="gpt-4o-mini", model_precision = torch.float16, system_eval:bool = False):
        super().__init__(model_name = model_name, model_params = model_params, model_precision = model_precision, system_eval = system_eval)
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        self.model_client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.model_name = model_name

    def preprocess_data(self, data_stream: torch.Tensor, prompt:Optional[str]=None):
        pil_image = self.to_pil_image(data_stream)
        base64_encoded_images = self.pil_to_base64(pil_image)
        return base64_encoded_images
    
    async def run_inference(self, data_stream: torch.Tensor, **kwargs):
        processed_inputs = self.preprocess_data(data_stream)
        messages = []
        if 'system_content' in kwargs:
            messages.append({"role": "system", "content": kwargs['system_content']})
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": kwargs['prompt']},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{processed_inputs}",
                        "detail": kwargs.get("detail", "auto")
                    }
                }
            ]
        })

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
            
            response = await self.model_client.chat.completions.create(**request_params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"API Error: {str(e)}"
