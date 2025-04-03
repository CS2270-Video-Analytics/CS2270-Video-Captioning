from config import Config
from ..model import Model
import torch
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import os 
from typing import Optional, Dict
from torchvision import transforms
from time import time
from torchvision import transforms

class GPTModel(Model):

    def __init__(self, model_name="gpt-4o-mini"):
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        OpenAI.api_key = os.environ['OPENAI_API_KEY']  # Set API key from environment
        self.model_client = OpenAI()
        self.model_name = model_name

        #auxilliary attributes for Tensor to image conversion
        self.to_pil_image = transforms.ToPILImage()
        self.model_name = Config.vision_model_name

    def preprocess_data(self, data_stream: torch.Tensor, prompt:Optional[str]=None):
        
        base64_encoded_images = self.convert_to_pil_image(data_stream)[0]

        return base64_encoded_images
    
    # Function to convert a PIL image to Base64
    def pil_to_base64(self, image:Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # Ensure a supported format like PNG or JPEG
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def convert_to_pil_image(self, torch_tensor:torch.Tensor):
        # Convert NumPy array to PIL Image
        return [self.pil_to_base64(self.to_pil_image(tensor)) for tensor in torch_tensor]
    
    def run_inference(self, data_stream: torch.Tensor, **kwargs):

        processed_inputs = self.preprocess_data(data_stream)

        info = {}

        messages = [
                    {"role": "system", "content": kwargs['system_content']},
                    {"role": "user", "content": [
                        {"type": "text", "text": kwargs['prompt']},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=kwargs['temperature'],
                top_p = kwargs['top_p'],
                max_tokens=kwargs['max_tokens'],
                frequency_penalty=kwargs['frequency_penalty'],
                presence_penalty=kwargs['presence_penalty'],
                stop_token=kwargs['stop_token']
            )
            info['error'] = None
        except openai.APIError as e:
            info['error'] = e

        return response.choices[0].message.content.strip(), info




