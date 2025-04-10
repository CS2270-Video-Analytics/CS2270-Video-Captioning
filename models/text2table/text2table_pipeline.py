import sys
import os
from config.config import Config
from typing import List
if Config.debug:
    import cv2
    import pdb
    from torchvision import transforms

from .OllamaText import OllamaText
import torch
import sqlite3

class Text2TablePipeline:

    def __init__(self, all_objects:List[str], all_captions:str=None):

        if Config.text2table_model == 'Seq2Seq':
            raise NotImplementedError("nrERROR: not implemented yet")
        elif Config.text2table_model == 'GPT3':
            self.text2table_incontext_prompt = Config.text2table_prompt
        elif Config.text2table_model == 'Ollama':
            self.text2table_incontext_prompt = Config.text2table_prompt
        
        #store prompt for per frame information extraction
        self.text2table_frame_prompt = Config.text2table_frame_prompt

        #store the attribute extraction prompt
        self.attribute_extraction_prompt = Config.text2table_attribute_extraction_prompt
        
        #initialize the model that needs to be used for captioning
        model_options = {'Ollama': OllamaText}
        assert Config.text2table_model in model_options, f'ERROR: model {Config.text2table_model} does not exist or is not supported yet'
        self.text2table_model = model_options[Config.text2table_model]()

        #extract the list of attributes to capture across objects
        self.object_attributes = ['video_id', 'frame_id', 'object', 'description', 'image_location', 'action']

        #store a list of unique objects to extract data for
        self.all_objects = all_objects
    
    def extract_attributes(self, all_captions:str):

        base_attributes = ['video_id', 'frame_id', 'object']

        frame_extraction_prompt = self.attribute_extraction_prompt.format(incontext_captions = Config.text2table_incontext_prompt, frame_captions = all_captions)

        extracted_attributes, __ = self.text2table_model.run_inference(query= frame_extraction_prompt)
        extracted_attributes = extracted_attributes.split('[')[1].split(']')[0].split(',')
        base_attributes += extracted_attributes

        return base_attributes

    
    def run_pipeline(self, caption: str, video_id:int, frame_id:int):

        #create the overall prompt structure
        if Config.text2table_model != 'Seq2Seq':
            text2table_frame_prompt = self.text2table_frame_prompt.format(formatted_schema = self.object_attributes, image_caption = caption, object_set = self.all_objects)
        else:
            text2table_frame_prompt = None
            raise NotImplementedError

        #generate the structured caption using text2table model
        structured_caption, info = self.text2table_model.run_inference(query = text2table_frame_prompt)

        #parse the structured caption as a partitioned of structured elements in JSON
        db_data_row = self.parse_table_output_t2t(structured_caption, video_id, frame_id)

        return db_data_row
    
    
    def parse_table_output_t2t(self, structured_caption:str, video_id:int, frame_id:int):

        
        parsed_rows = structured_caption.strip().split("<n>")
        
        # Filter out empty rows and split remaining rows by <t>
        parsed_rows = [[video_id, frame_id] + row.split("<t>") for row in rows if parsed_rows.strip()]
        
        return parsed_rows



if __name__ == '__main__':
    pdb.set_trace()

    test_caption = "The image shows a city street with cars and buildings.\
The foreground features a white car with a yellow license plate, facing away from the camera. The license plate reads \"54-628-74\" and has a yellow background with black text.\
A white car is driving on the left side of the road, heading towards the camera.\
A silver car is driving on the right side of the road, heading away from the camera.\
A white van is driving on the left side of the road, heading towards the camera.\
A green traffic light is visible on the left side of the road, indicating that it is currently green and allowing traffic to proceed.\
A pedestrian is crossing the street on the right side of the road, using a crosswalk.\
There are several trees and palm trees along the street, providing shade and a natural environment.\
Buildings line the street, including a row of tall, beige buildings on the left side and a row of shorter, tan buildings on the right side.\
The sky is blue and clear, indicating a sunny day.\
The image captures a typical urban scene, with cars and pedestrians navigating through the city streets, surrounded by trees and"

    all_objects = ['car','van','traffic light', 'pedestrian', 'street', 'tree']
    
    #create text 2 table pipeline
    t2t = Text2TablePipeline(all_objects)

    #dummy run with video and frame id 1
    db_data_row = t2t.run_pipeline(test_caption, 1, 1)
    