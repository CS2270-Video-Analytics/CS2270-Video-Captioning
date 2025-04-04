import sys
sys.path.append('..')
from config import Config
import os
import io
import base64
from ..model import Model
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

class OllamaText(LanguageModel):
    def __init__(self, model_name: str, model_params: Dict = None):
        self.model_name = model_name
        self.model_params = model_params
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
    
    def run_inference(self, query: str, **kwargs):
        print("Starting inference...")

        #additional return values in a dictionary
        info = {}
        outputs = None

        if self.model_params['system_eval']:
            start_time = time.now()

        print("Sending to Ollama...")
        with torch.no_grad():
            try:
                outputs = ollama.chat(
                    model= self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': query,
                    }],
                    options = {
                        'top_k': self.model_params['top_k'],
                        'top_p': self.model_params['top_p'],
                        'temperature': self.model_params['temperature'],
                        'repeat_penalty': self.model_params['repeat_penalty'],
                        'presence_penalty': self.model_params['presence_penalty'],
                        'frequency_penalty': self.model_params['frequency_penalty'],
                        'stop': self.model_params['stop_token'],
                        'max_tokens': self.model_params['max_tokens'],
                        'batch_size': self.model_params['batch_size'],
                        'num_threads': self.model_params['num_threads'],
                    })
                print(f"Received response from Ollama")
                outputs = outputs.message.content
            except Exception as e:
                print(f"Error in Ollama inference: {e}")
                info['error'] = e
                outputs = "Error generating caption"
        
        if self.model_params['system_eval']:
            end_time = time.now()
            elapsed = end_time - start_time
            info['time'] = elapsed

        return outputs, info
    
    def preprocess_data(self, data_stream: torch.Tensor, prompt: Optional[str]=None):
        pass
