import sys
import os

from collections import deque
import torch

from models.vision_language_models.BLIP import BLIP
from models.vision_language_models.BLIP2 import BLIP2
from models.vision_language_models.CLIP import CLIP
from models.vision_language_models.OllamaVision import OllamaVision
from models.vision_language_models.OpenAIVision import OpenAIVision
from config.config import Config



if Config.debug:
    import cv2
    import pdb
    from torchvision import transforms

from models.vision_language_models.BLIP import BLIP
from models.vision_language_models.BLIP2 import BLIP2
from models.vision_language_models.CLIP import CLIP
from models.vision_language_models.OllamaVision import OllamaVision
from models.vision_language_models.OpenAIVision import OpenAIVision
import torch

class CaptioningPipeline():

    def __init__(self):
        #construct prompts to be used in captioning pipeline
        if Config.previous_frames:
            self.description_prompt = Config.sliding_window_caption_prompt_format
        else:
            self.description_prompt = Config.question_prompt_format.format(question = Config.generic_caption_prompt_format)

        if Config.obj_focus:
            self.object_prompt = Config.object_extraction_prompt_format
            self.object_set = set(Config.init_object_set)
        
        #construct prompt specific for OpenAI system context
        self.context_prompt = Config.captioning_context_prompt_format

        #track the window of previous image descriptions
        self.previous_descriptions = deque()
        self.previous_descriptions.append('')
        self.previous_descriptions.append('The video starts with a black screen')
        self.sliding_window_size = Config.sliding_window_size

        #initialize the model that needs to be used for captioning
        model_options = {'OllamaVision': OllamaVision, 'BLIP': BLIP, 'BLIP2': BLIP2, 'OpenAI':OpenAIVision}
        [caption_model, caption_model_name] = Config.caption_model_name.split(';')
        assert caption_model in model_options, f'ERROR: model {Config.caption_model_name} does not exist or is not supported yet'

        self.caption_model = model_options[caption_model](model_name = caption_model_name, model_params = Config.caption_model_params, model_precision=Config.model_precision, system_eval=Config.system_eval)

        #create models for clip vector embeddings
        self.clip_model = CLIP(model_name = Config.clip_model_name, model_precision=Config.model_precision, system_eval=Config.model_precision)

    def run_pipeline(self, data_stream: torch.Tensor, video_id:int, frame_id:int):
        
        #(1) add the previous frame description to the prompt
        if Config.previous_frames:
            previous_frames_descriptions = '\n-'.join(self.previous_descriptions)
            description_prompt = self.description_prompt.format(object_set = ','.join(self.object_set))
        else:
            description_prompt = self.description_prompt

        #(2) pass the model the captioning prompt for captioning
        #TODO: figure out data streaming
        description, info = self.caption_model.run_inference(data_stream = data_stream, **dict(prompt = description_prompt, system_content = self.context_prompt))
        
        #(3) process the new description: append into previous queue + pop out from queue if needed
        self.previous_descriptions.append(description)
        if len(self.previous_descriptions) > self.sliding_window_size:
            self.previous_descriptions.popleft()
        
        #(4) generate a set of new objects in the current frame and add to the self.object_set
        if Config.obj_focus:
            obj_prompt = self.object_prompt.format(curr_img_caption = self.previous_descriptions[-1], object_set = ','.join(self.object_set))
            new_objs, info = self.caption_model.run_inference(data_stream = data_stream, **dict(prompt = obj_prompt))
            new_objs = new_objs[1:-1].split(',')
            new_objs = [obj.strip().lower() for obj in new_objs]
            self.object_set.update(new_objs)

        #(5) generate clip embedding
        image_embedding, info = self.clip_model.run_inference(image).detach().cpu()

        return [video_id, frame_id, description, self.object_set, image_embedding]
        

if __name__ == '__main__':

    pdb.set_trace()
    captioner = CaptioningPipeline()

    image_base_path = os.path.relpath('./datasets/mock_caption_data/bdd')

    for i, img in enumerate(os.listdir(image_base_path)):

        # Load image using OpenCV
        image_path = os.path.join(image_base_path, img)
        image = cv2.imread(image_path)  # Shape: (H, W, C), BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Convert to PyTorch tensor
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        
        vid_id, frame_id, descrip, image_embed = captioner.run_pipeline(data_stream = image_tensor, video_id = 0, frame_id=i)
        