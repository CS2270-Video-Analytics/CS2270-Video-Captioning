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
    
    @abstractmethod
    def run_inference(self, data_stream:"Tensor", **kwargs):
        pass

    #NOTE: optional to override, not all models have to be finetuned
    def finetune_model(self, dataloader: "DataLoader"):
        pass

