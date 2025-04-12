import sys
import os
from config.config import Config
from typing import List, Dict
import re
import json

if Config.debug:
    import cv2
    import pdb
    from torchvision import transforms
import pdb
from models.language_models.Anthropic import Anthropic
from models.language_models.OpenAIText import OpenAIText
from models.language_models.OllamaText import OllamaText
from models.language_models.DeepSeek import DeepSeek
import torch
import sqlite3

class Text2TablePipeline():
    def __init__(self, all_objects:List[str]):
        self.text2table_frame_prompt = Config.text2table_frame_prompt

        #store all prompts for text2table
        self.attribute_extraction_prompt = Config.text2table_attribute_extraction_prompt
        self.schema_extraction_prompt_format = Config.text2table_schema_generation_prompt

        #initialize the model that needs to be used for captioning
        model_options = {'Ollama': OllamaText, 'OpenAI': OpenAIText, 'Anthropic': Anthropic, 'DeepSeek':DeepSeek}
        [text2table_model, text2table_model_name] = Config.caption_model_name.split(';')
        assert text2table_model in model_options, f'ERROR: model {text2table_model} does not exist or is not supported yet'
        
        self.text2table_model = model_options[text2table_model](model_params = Config.text2table_params, model_name=text2table_model_name, model_precision=Config.model_precision, system_eval=Config.system_eval)

        #extract the list of attributes to capture across frame descriptions
        self.table_attributes = []

        #store a list of unique objects to extract data for
        self.all_objects = all_objects
    
    def clear_pipeline(self):
        #clear the cache that remains for previous runs of text2table
        self.table_attributes = []
        self.all_objects = []

    def update_objects(self, all_objects:List[str]):
        self.all_objects = all_objects

    def extract_table_attributes(self, all_captions:str):

        frame_extraction_prompt = self.attribute_extraction_prompt.format(incontext_examples = Config.text2table_incontext_prompt, all_joined_captions = all_captions)
        extracted_attributes, __ = self.text2table_model.run_inference(data_stream= frame_extraction_prompt)

        return extracted_attributes.strip()
    
    def extract_table_schemas(self, all_captions:str):

        extracted_attributes = self.extract_table_attributes(all_captions)

        schema_extraction_prompt_format = self.schema_extraction_prompt_format.format(attributes=extracted_attributes)
        generated_schemas, __ = self.text2table_model.run_inference(data_stream = schema_extraction_prompt_format)
        generated_schemas = self.clean_schema(generated_schemas)

        return generated_schemas
    
    def build_json_template(self, schema_dict, frame_id_placeholder="{frame_id}"):
        pdb.set_trace()
        lines = []
        for table, cols in schema_dict.items():
            lines.append(f'  "{table}": [')
            lines.append("    {")
            for col in cols:
                if col == "frame_id":
                    value = f'"{frame_id_placeholder}"'
                else:
                    value = '...'
                lines.append(f'      "{col}": {value},')
            lines[-1] = lines[-1].rstrip(",")  # remove trailing comma
            lines.append("    }")
            lines.append("  ],")
        lines[-1] = lines[-1].rstrip(",")  # remove trailing comma
        return "\n".join(lines)
    

    def clean_schema(self, schema: str) -> str: 
        lines = schema.strip().splitlines() 
        cleaned_lines = [line for line in lines if not re.match(r"^\s*```.*$", line)] 
        return "\n".join(cleaned_lines)


    def parse_db_schema(self, schema_str):
        pdb.set_trace()
        table_defs = {}
        current_table = None

        for line in schema_str.strip().splitlines():
            line = line.strip()
            if line.startswith("Table:"):
                current_table = line.split("Table:")[1].strip()
                table_defs[current_table] = []
            elif line.startswith("-") and current_table:
                match = re.match(r"-\s*(\w+)\s*\(", line)
                if match:
                    column_name = match.group(1)
                    table_defs[current_table].append(column_name)

        return table_defs

    def extract_json_from_response(self, text:str):
        # Strip markdown-style code block if present
        if "```" in text:
            text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        return text.strip()
    
    def run_pipeline_video(self, video_data:List[tuple], database_schema:str):
        pdb.set_trace()
        # Parse schema to extract table and column structure
        parsed_db_schema = self.parse_db_schema(database_schema)
        json_schema_template = self.build_json_template(schema_dict=parsed_db_schema)
        
        #collate all data for all frames before inserting
        frame_data = []

        #iterate all rows from the video data raw captions and run the pipeline per batch
        for i, (video_id, frame_id, caption, __) in enumerate(video_data):

            frame_obj_data = self.run_pipeline(parsed_db_schema=parsed_db_schema, caption = caption, video_id = video_id, frame_id = frame_id)
            frame_data += frame_obj_data

            if len(frame_data) >= Config.batch_size or i == len(video_data) - 1: 
                yield frame_data
                frame_data = [] #empty the batch
                
    def run_pipeline(self, parsed_db_schema:Dict, caption: str, video_id:int, frame_id:int):
        pdb.set_trace()
        try:
            raw_response, _ = self.text2table_model.run_inference(data_stream= self.text2table_frame_prompt.format(caption=caption, object_set=self.all_objects, schema=json_schema_template, frame_id = frame_id))
            json_response = self.extract_json_from_response(raw_response)
            json_response = json.loads(json_response)

            db_data_rows = []

            for table, columns in parsed_db_schema.items():
                records = json_response.get(table, [])
                if not isinstance(records, list):
                    continue

                for record in records:
                    table_vals = [record.get(col) for col in columns]
                    db_data_rows.append(table_vals)
        except Exception as e:
            raise f"ERROR: Cannot process video {video_id} and frame {frame_id} - {e}"

        return db_data_rows
    
    
    def parse_table_output_t2t(self, structured_caption:str, video_id:int, frame_id:int):
        #NOTE: currently this is unused, but if we need text2table we can use this
        # parsed_rows = structured_caption.strip().split("<r>")
        # parsed_rows = [[video_id, frame_id] + row.strip().split("<c>") for row in parsed_rows if parsed_rows]
        # parsed_rows = parsed_rows[1:-1]
        
        parsed_rows = [list(map(str.strip, row.split("<c>")))[1:-1] for row in structured_caption.split("<r>") if row.strip()]
        
        parsed_rows_test = [list(map(str.strip, row.split("<c>"))) for row in structured_caption.split("<r>") if row.strip()]
        
        #parse to ensure added rows have right number of columns
        parsed_rows = [[video_id, frame_id] + row for row in parsed_rows if len(row)==len(self.table_attributes)]
        
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











