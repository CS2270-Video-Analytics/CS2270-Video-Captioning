from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Import the type-only dependencies
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    import numpy as np
    from torch import Tensor

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def preprocess_data(self, data_stream:"Tensor", text_input:Optional[str]):
        pass

class VisionLanguageModel(Model, ABC):
    def __init__(self):
        super().__init__()
        self.to_pil_image = transforms.ToPILImage()
        
    
    @abstractmethod
    def run_inference(self, data_stream:"Tensor", **kwargs):
        pass

    #NOTE: optional to override, not all models have to be finetuned
    def finetune_model(self, dataloader: "DataLoader"):
        pass
    
    def convert_to_pil_image(self, torch_tensor:torch.Tensor):
        # Convert NumPy array to PIL Image
        return self.to_pil_image(torch_tensor)

    # Function to convert a PIL image to Base64
    def pil_to_base64(self, image:Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # Ensure a supported format like PNG or JPEG
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    

class LanguageModel(Model, ABC):

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def run_inference(self, query:str, **kwargs):
        pass

    #NOTE: optional to override, not all models have to be finetuned
    def finetune_model(self, dataloader: "DataLoader"):
        pass