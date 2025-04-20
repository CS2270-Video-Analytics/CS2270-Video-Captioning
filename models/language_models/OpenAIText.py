from config import Config
from ..model import LanguageModel
import torch
from dotenv import load_dotenv, find_dotenv
import openai
from openai import AsyncOpenAI
import os 
from typing import Optional, Dict
from torchvision import transforms
from time import time
import asyncio


class OpenAIText(LanguageModel):
    def __init__(self, model_params: Dict, model_name:str="gpt-4o-mini", model_precision = torch.float16, system_eval:bool = False):
        super().__init__(model_params=model_params, model_precision=model_precision, system_eval=system_eval)
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        self.model_client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.model_name = model_name

    def preprocess_data(self, data_stream: str, text_input:Optional[str]):
        pass
    
    async def run_inference(self, data_stream: str, **kwargs):
        messages = [
            {"role": "user", "content": data_stream}
        ]
        if 'system_content' in kwargs:
            messages.append({"role": "system", "content": kwargs['system_content']})
        
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
        
        delay = 0.2
        max_retries = 5
        for _ in range(max_retries):
            try:
                response = await self.model_client.chat.completions.create(**request_params)
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "rate_limit_exceeded" in str(e) or "insufficient_quota" in str(e):
                    print(f"Rate limit exceeded, retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    return f"API Error: {str(e)}"
        return "Max retries ({max_retrie} times) exceeded"
