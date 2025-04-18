from abc import ABC, abstractmethod
from PIL import Image
import io
import base64
import torch
from torchvision import transforms
from typing import TYPE_CHECKING, Optional, Union, Dict


if TYPE_CHECKING:
    # Import the type-only dependencies
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    import numpy as np
    from torch import Tensor

class Model(ABC):
    def __init__(self, model_params = {}, model_precision = torch.float16, system_eval:bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_precision =  model_precision
        self.system_eval = system_eval
        self.model_params = model_params #store the parameters to the model

    @abstractmethod
    def preprocess_data(self, data_stream:Union["Tensor",str], text_input:Optional[str]):
        pass

    @abstractmethod
    def run_inference(self, data_stream:Union["Tensor",str], **kwargs):
        pass

class VisionLanguageModel(Model):
    def __init__(self, model_params:dict, model_name: str, model_precision = torch.float16, system_eval:bool = False):
        super().__init__(model_params = model_params, model_precision = model_precision, system_eval = system_eval)
        self.to_pil_image = transforms.ToPILImage()
        
    #NOTE: optional to override, not all models have to be finetuned
    def finetune_model(self, dataloader: "DataLoader"):
        pass
    
    def convert_to_pil_image(self, torch_tensor:torch.Tensor):
        # Convert NumPy array to PIL Image
        return self.to_pil_image(torch_tensor)

    # Function to convert a PIL image to Base64
    def pil_to_base64(self, image:Image):
        """
        Convert a PIL image to a base64 encoded string.
        
        Args:
            image (PIL.Image): The PIL image to convert
            
        Returns:
            str: Base64 encoded string of the image
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # Ensure a supported format like PNG or JPEG
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    

class LanguageModel(Model):
    def __init__(self, model_params: Dict, model_precision = torch.float16, system_eval: bool = False):
        super().__init__(model_params=model_params, model_precision=model_precision, system_eval=system_eval)

    #NOTE: optional to override, not all models have to be finetuned
    def finetune_model(self, dataloader: "DataLoader"):
        pass