from models.model import VisionLanguageModel
import torch
from typing import Optional, Dict
import clip
import time


class CLIP(VisionLanguageModel):

    def __init__(self, model_params:dict = {}, model_name:int="ViT-L/14", model_precision = torch.float16, system_eval:bool = False):
        super().__init__(model_name = model_name, model_params = model_params, model_precision = model_precision, system_eval = system_eval)
        self.model_name = model_name
        
        #auxilliary attributes for Tensor to image conversion
        self.clip_model, self.clip_preprocess = clip.load(self.model_name, device=self.device)



    def preprocess_data(self, data_stream: torch.Tensor, prompt:Optional[str]=None):
        
        pil_image = self.to_pil_image(data_stream)
        image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

        return image
    
    
    def run_inference(self, data_stream: torch.Tensor,  **kwargs):

        processed_inputs = self.preprocess_data(data_stream)

        info = {}

        if self.system_eval:
            start_time = time.time()

        try:
            with torch.no_grad():
                image_embedding = self.clip_model.encode_image(processed_inputs)
            info['error'] = None
        except Exception as e:
            info['error'] = e

        if self.system_eval:
            end_time = time.time()
            elapsed = end_time - start_time
            info['time'] = elapsed


        return image_embedding, info




