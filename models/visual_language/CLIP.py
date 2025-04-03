from config import Config
from ..model import VisualLanguageModel
import torch
from typing import Optional, Dict

class CLIP(VisualLanguageModel):

    def __init__(self, model_name="ViT-L/14"):
       
        
        self.model_name = Config.clip_model_name if model_name is None else model_name

        #auxilliary attributes for Tensor to image conversion
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(Config.clip_model_name, device=self.device)

        #auxilliary for preprocessing
        self.to_pil_image = transforms.ToPILImage()


    def preprocess_data(self, data_stream: torch.Tensor, prompt:Optional[str]=None):
        
        pil_image = self.to_pil_image(data_stream)
        image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

        return image
    
    
    def run_inference(self, data_stream: torch.Tensor, **kwargs):

        processed_inputs = self.preprocess_data(data_stream)

        info = {}

        try:
            with torch.no_grad():
                image_embedding = self.clip_model.encode_image(processed_inputs)
            info['error'] = None
        except Exception as e:
            info['error'] = e

        return image_embedding, info




