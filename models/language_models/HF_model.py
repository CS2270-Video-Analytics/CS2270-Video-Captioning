import sys
sys.path.append('..')
from config import Config
import os
import io
import base64
from ..model import LanguageModel
import ollama
from PIL import Image
import torch

from typing import Optional, Dict
from time import time
from torchvision import transforms
import requests
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

class HuggingFaceText(LanguageModel):
    def __init__(self, model_params: Dict, model_name: str = "meta-llama/Llama-2-7b-chat-hf", model_precision = torch.float16, system_eval: bool = False):
        super().__init__(model_params=model_params, model_precision=model_precision, system_eval=system_eval)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_precision,
            device_map="auto"
        )

    def preprocess_data(self, data_stream: str, text_input: Optional[str]):
        pass
    
    def run_inference(self, data_stream: str, **kwargs):
        info = {}
        
        if self.system_eval:
            start_time = time.time()

        try:
            # Prepare input
            messages = [
                {"role": "system", "content": kwargs['system_content']},
                {"role": "user", "content": data_stream}
            ]
            
            # Format messages for the model
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_params['max_tokens'],
                    temperature=self.model_params['temperature'],
                    top_p=self.model_params['top_p'],
                    do_sample=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            info['error'] = None
            
        except Exception as e:
            info['error'] = e
            response = "Error generating response"
        
        if self.system_eval:
            end_time = time.time()
            elapsed = end_time - start_time
            info['time'] = elapsed

        return response, info