import sys
import os
from config.config import Config
from collections import deque
import clip

if Config.debug:
    import cv2
    import pdb
    from torchvision import transforms

from .BLIP import BLIP
from .BLIP2 import BLIP2
from .LLamaVision import LLamaVision
import torch

class CaptioningPipeline():

    def __init__(self, video_id: int):
        
        if Config.previous_frames:
            self.description_prompt = Config.sliding_window_caption_prompt_format
        else:
            self.description_prompt = Config.question_prompt_format.format(question = Config.generic_caption_prompt_format)

        if Config.obj_focus:
            self.object_prompt = Config.object_extraction_prompt_format
            self.object_list = []
        
        #track the window of previous image descriptions
        self.previous_descriptions = deque()
        self.previous_descriptions.append('')
        self.previous_descriptions.append('The video starts with a black screen')
        self.sliding_window_size = Config.sliding_window_size

        #initialize the model that needs to be used for captioning
        model_options = {'LLamaVision': LLamaVision, 'BLIP': BLIP, 'BLIP2': BLIP2}
        assert Config.caption_model in model_options, f'ERROR: model {Config.caption_model} does not exist or is not supported yet'
        self.caption_model = model_options[Config.caption_model]()

        #create models for clip vector embeddings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(Config.clip_model_name, device=self.device)

    def update_video_id(self, new_video_id:int):
        self.video_id = new_video_id

    def run_pipeline(self, data_stream: torch.Tensor, video_id:int, frame_id:int):

        #add the previous frame description to the prompt
        if Config.previous_frames:
            previous_frames_descriptions = '\n-'.join(self.previous_descriptions)
            description_prompt = self.description_prompt.format(prev_frames_context = previous_frames_descriptions)
        else:
            description_prompt = self.description_prompt

        #pass the model the captioning prompt for captioning
        #TODO: figure out data streaming
        description, info = self.caption_model.run_inference(data_stream = data_stream, **dict(prompt = description_prompt))
        
        #process the new description: append into previous queue + pop out from queue if needed
        self.previous_descriptions.append(description)
        if len(self.previous_descriptions) > self.sliding_window_size:
            self.previous_descriptions.popleft()
        
        #generate a set of new objects in the current frame and add to the self.object_list
        if Config.obj_focus:
            obj_prompt = self.object_prompt.format(self.previous_descriptions[-1], ','.join(self.object_list))
            new_objs, info = self.caption_model.run_inference(data_stream = data_stream, kwargs = dict(prompt = description_prompt))

            self.object_list.append(new_objs)

        # generate clip embedding
        pil_image = self.caption_model.to_pil_image(data_stream)
        image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(image)

        #TODO: None to be replaced with object list
        return [video_id, frame_id, description, None, image_embedding]
        

if __name__ == '__main__':

    pdb.set_trace()
    captioner = CaptioningPipeline(video_id=1)

    image_base_path = os.path.relpath('../../datasets/mock_caption_data/bdd')

    for img in os.listdir(image_base_path):

        # Load image using OpenCV
        image_path = os.path.join(image_base_path, img)
        image = cv2.imread(image_path)  # Shape: (H, W, C), BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Convert to PyTorch tensor
        transform = transforms.ToTensor()
        image_tensor = transform(image)

        vid_id, frame_id, descrip, image_embed = captioner.run_pipeline(data_stream = image_tensor, video_id = 0)
        