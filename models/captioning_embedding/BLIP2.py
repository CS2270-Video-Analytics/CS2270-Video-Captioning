import sys
sys.path.append('..')

import os
from config import Config
from ..model import Model
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

from typing import Optional, Dict
from torchvision import transforms
from time import time
if Config.debug:
    import pdb
#TODO: find a way to batch process and parallelize processing of frames from many videos
#TODO: object extraction - siteratively update set of objects in the video across frames

class BLIP2(Model):
    
    def __init__(self):

        #attributes from configs 
        self.precision = Config.model_precision

        # Load BLIP-2 model and processor
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=self.precision)
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)

        self.system_eval = Config.system_eval
    
        #auxilliary attributes for Tensor to image conversion
        self.to_pil_image = transforms.ToPILImage()

    def convert_to_pil_image(self, torch_tensor:torch.Tensor):
        # Convert NumPy array to PIL Image
        return [self.to_pil_image(tensor) for tensor in torch_tensor]

    def preprocess_data(self, data_stream: torch.Tensor, prompt:Optional[str]):
        
        images = self.convert_to_pil_image(data_stream)

        return self.processor(images=images, text = prompt, return_tensors="pt").to(self.device, self.precision)
    
    
    def run_inference(self, data_stream: torch.Tensor, **kwargs):

        #additional return values in a dictionary
        info = {}

        if self.system_eval:
            start_time = time.now()

        self.model.eval()
        processed_inputs = self.preprocess_data(data_stream, kwargs['prompt'])

        # Generate caption
        with torch.no_grad():

            try:
                output_ids = self.model.generate(**processed_inputs)
                outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)
                outputs = outputs.split(':')[2].strip()
            except Exception as e:
                info['error'] = e
        
        if self.system_eval:
            end_time = time.now()
            elapsed = end_time - start_time
            info['time'] = elapsed

        return outputs, info


if __name__ == '__main__':

    import pdb

    obj_list = []
    kwargs = {'prev_frame_context': '-A blank screen', 
            'objs': obj_list, 
            'obj_focus':False, 
            'text_input': "Current Image Description:\n",
            'prev_frames':False
            }
    
    num_prev_description = 3

    curr_frame = 1
    output_result = []

    

    model = BLIP2(model_configs={'model_precision':torch.float16, 
                                'system_eval':False, 
                                'fuckoff': 'Using the descriptions of the previous images of a video, generate a detailed description of the current image. Describe all entities and their relationship with each other in detail\nDescription of previous image sequence:\n{prev_frames_context}', 
                                'template_prompt': 'Generate a detailed description of the current image. Describe all objects in the image and their relationship with each other in detail', 

                                'obj_extraction_prompt': 'Given the generated prompt and list of objects, identify any new objects not in the object list:', 
                                'obj_prompt': 'Objects in previous frame sequence:\n[{prev_objs}]'
                                })
    
    import sys
    sys.path.append('../..')
    import os

    #for sample example, extract concurrent frames as a batched tensor and give to inference
    base_path = '../../datasets/mock_caption_data'
    data_path = os.listdir(base_path)
    transform = transforms.ToTensor()

    for img in data_path:

        #load the model and process it as a Torch tensor
        image = Image.open(os.path.join(base_path, img))
        tensor_image = transform(image).unsqueeze(0)
        
        pdb.set_trace()
        captions, info = model.run_inference(data_stream = tensor_image, prev_frame_context=kwargs['prev_frame_context'], objs=kwargs['objs'], obj_focus=kwargs['obj_focus'], text_input=kwargs['text_input'], prev_frames=kwargs['prev_frames'])

        #update the kwargs assuming that video processed frame by frame
        kwargs['prev_frame_context'] = '\n-'.join(output_result[min(len(output_result)-num_prev_description,0):len(output_result)])

        #aggregate results
        output_result.append(captions[0])
        curr_frame += 1

    print("Generated Caption: ", captions)