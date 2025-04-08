import sys
sys.path.append('..')
import os
import io
import base64
from ..model import VisualLanguageModel
import ollama
from PIL import Image
import torch

from typing import Optional, Dict
from torchvision import transforms
from time import time
from torchvision import transforms
import requests
import subprocess


#TODO: find a way to batch process and parallelize processing of frames from many videos
#TODO: object extraction - siteratively update set of objects in the video across frames

class OllamaVision(VisualLanguageModel):
    
    def __init__(self, model_params:dict, model_name: str, model_precision = torch.float16, system_eval:bool = False):
        super().__init__(model_params = model_params, model_precision = model_precision, system_eval = system_eval)
        
        #auxilliary attributes for Tensor to image conversion
        self.model_name = model_name

        #assert that LLaMa model is running on another terminal (sanity check)
        assert self.is_ollama_model_running(), "ERROR: Ollama is not running. Run it on another terminal first"

        


    def get_gpu_processes(self):
        # Run nvidia-smi to get the list of GPU processes
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        processes = result.stdout.decode().splitlines()
        return processes
    
    def is_ollama_model_running(self):
        try:
            # Try to connect to the system-wide Ollama service
            response = requests.get('http://127.0.0.1:11434/api/version')
            if response.status_code == 200:
                print("Successfully connected to Ollama service")
                return True
            return False
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return False
    

    def preprocess_data(self, data_stream: torch.Tensor, prompt:Optional[str]=None):
        
        base64_encoded_images = self.pil_to_base64(self.convert_to_pil_image(data_stream))

        return base64_encoded_images
    
    
    def run_inference(self, data_stream: torch.Tensor,  **kwargs):
        print("Starting inference...")

        #additional return values in a dictionary
        info = {}
        outputs = None

        if self.system_eval:
            start_time = time.time()

        print("Processing inputs...")
        processed_inputs = self.preprocess_data(data_stream)

        print("Sending to Ollama...")
        with torch.no_grad():
            try:
                outputs = ollama.chat(
                    model= self.model_name,
                    messages=[{
                        'role': 'user',
                        'content':kwargs['prompt'],
                        'images': [processed_inputs]
                    }],
                    keep_alive = self.model_params['keep_alive'],
                    options={
                        'temperature': self.model_params['temperature'],
                        'top_k': self.model_params['top_k'],
                        'top_p': self.model_params['top_p'],
                        'num_ctx': self.model_params['num_ctx'],
                        'repeat_penalty': self.model_params['repeat_penalty'],
                        'presence_penalty': self.model_params['presence_penalty'],
                        'frequency_penalty': self.model_params['frequency_penalty'],
                        'num_predict': self.model_params['max_tokens'],
                        'stop': self.model_params['stop_tokens']
                    }
                    )
                print(f"Received response from Ollama")
                outputs = outputs.message.content
            except Exception as e:
                print(f"Error in Ollama inference: {e}")
                info['error'] = e
                outputs = "Error generating caption"
        
        if self.system_eval:
            end_time = time.time()
            elapsed = end_time - start_time
            info['time'] = elapsed

        return outputs, info
