import sys
sys.path.append('..')
from config import Config
import os
import io
import base64
from model import Model
import ollama
from PIL import Image
import torch

from typing import Optional, Dict
from torchvision import transforms
from time import time
from torchvision import transforms
if Config.debug:
    import pdb
import requests


#TODO: find a way to batch process and parallelize processing of frames from many videos
#TODO: object extraction - siteratively update set of objects in the video across frames

class LLamaVision(Model):
    
    def __init__(self):
        
        #attributes from configs 
        self.precision = Config.model_precision
        self.system_eval = Config.system_eval
        
        #auxilliary attributes for Tensor to image conversion
        self.to_pil_image = transforms.ToPILImage()
        self.model_name = Config.llama_model_name

        #assert that LLaMa model is running on another terminal (sanity check)
        assert self.is_ollama_running(), "ERROR: Ollama is not running. Run it on another terminal first"

        #auxilliary attributes for Tensor to image conversion
        self.to_pil_image = transforms.ToPILImage()

    def is_ollama_running(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
    
    # Function to convert a PIL image to Base64
    def pil_to_base64(self, image:Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # Ensure a supported format like PNG or JPEG
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def convert_to_pil_image(self, torch_tensor:torch.Tensor):
        # Convert NumPy array to PIL Image
        return [self.pil_to_base64(self.to_pil_image(tensor)) for tensor in torch_tensor]

    def preprocess_data(self, data_stream: torch.Tensor, prompt:Optional[str]=None):
        
        base64_encoded_images = self.convert_to_pil_image(data_stream)[0]

        return base64_encoded_images
    
    
    def run_inference(self, data_stream: torch.Tensor, **kwargs):

        #additional return values in a dictionary
        info = {}

        if self.system_eval:
            start_time = time.now()
        
        processed_inputs = self.preprocess_data(data_stream)

        pdb.set_trace()
        # Generate caption
        with torch.no_grad():
            
            try:
                outputs = ollama.chat( model= self.model_name,
                                        messages=[{
                                            'role': 'user',
                                            'content':kwargs['prompt'],
                                            'images': [processed_inputs],
                                            'max_tokens': Config.max_tokens
                                        }])
                outputs = outputs.message.content

            except Exception as e:
                info['error'] = e
        
        if self.system_eval:
            end_time = time.now()
            elapsed = end_time - start_time
            info['time'] = elapsed

        return outputs, info






