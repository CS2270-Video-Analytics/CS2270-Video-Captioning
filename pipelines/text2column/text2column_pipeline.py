from config.config import Config
from typing import List, Dict
from models.language_models.Anthropic import AnthropicText
from models.language_models.OpenAIText import OpenAIText
from models.language_models.OllamaText import OllamaText
from models.language_models.DeepSeek import DeepSeek
from models.vision_language_models.OpenAIVision import OpenAIVision
from models.vision_language_models.BLIP import BLIP
from models.vision_language_models.DeepSeek import DeepSeek
from models.vision_language_models.OllamaVision import OllamaVision
import sqlite3
import asyncio
import pdb
import json
import re

class Text2ColumnPipeline():
    def __init__(self):

        #store all prompts for text2col
        self.attribute_extraction_prompt = Config.text2col_raw_extraction_prompt
        self.structured_attribute_extraction_prompt = Config.text2col_structured_extraction_prompt

        #initialize the model that needs to be used for captioning
        language_model_options = {'Ollama': OllamaText, 'OpenAI': OpenAIText, 'Anthropic': AnthropicText, 'DeepSeek':DeepSeek}
        [text2colmn_language_model, text2colmn_language_model_name] = Config.text2col_model_name.split(';')
        assert text2colmn_language_model in language_model_options, f'ERROR: model {text2colmn_language_model} does not exist or is not supported yet'
        
        vision_language_model_options = {'BLIP': BLIP, 'OpenAI': OpenAIVision, 'DeepSeek': DeepSeek, 'OllamaVision':OllamaVision}
        [text2colmn_vl_model, text2colmn_vl_model_name] = Config.text2col_model_name.split(';')
        assert text2colmn_vl_model in vision_language_model_options, f'ERROR: model {text2colmn_vl_model} does not exist or is not supported yet'
        

        self.text2column_vision_model = vision_language_model_options[text2colmn_vl_model](
                                    model_params = Config.text2column_params, 
                                    model_name=text2colmn_vl_model_name, 
                                    model_precision=Config.model_precision, 
                                    system_eval=Config.system_eval)
        self.text2column_text_model = language_model_options[text2colmn_language_model](
                                    model_params = Config.text2column_params, 
                                    model_name=text2colmn_language_model_name, 
                                    model_precision=Config.model_precision, 
                                    system_eval=Config.system_eval)
        self.caption_detail = Config.caption_detail
    
    async def process_new_attributes_tuple(self, extracted_structured_attributes: str, new_attributes: list):
        # Find the first {...} block
        dict_match = re.search(r'\{.*?\}', extracted_structured_attributes, re.DOTALL)
        if not dict_match:
            raise ValueError("No dictionary found in the string.")

        dict_str = dict_match.group()

        # Remove braces and split into key-value pairs
        inner_content = dict_str.strip('{}').strip()
        key_value_pairs = re.split(r',\s*', inner_content)

        # Build a simple dictionary
        attr_dict = {}
        for pair in key_value_pairs:
            if ':' in pair:
                key, value = pair.split(':', 1)
                key = key.strip()
                value = value.strip()
                attr_dict[key] = value

        # Extract in the order of all_attributes
        extracted = tuple(attr_dict.get(attr) for attr in new_attributes)

        return extracted
    
    async def generate_new_column_data(self, table_name: str, frame, table_rows: list, table_columns: list, new_attributes: list):
        batch_rows= []
        for row in table_rows:
            all_new_attributes = ','.join(new_attributes)
            current_attributes = {k:v for (k,v) in zip(table_columns, row)}
            curr_obj_attributes = {k: v for k, v in current_attributes.items() if k not in ['video_id', 'frame_id']}
            pdb.set_trace()
            attribute_extraction_prompt = self.attribute_extraction_prompt.format(table_name = table_name, current_attributes = str(curr_obj_attributes), all_new_attributes=all_new_attributes)
            extracted_raw_attributes = await self.text2column_vision_model.run_inference(frame, **dict(detail=self.caption_detail, prompt = attribute_extraction_prompt))

            # structured_attribute_extraction_prompt = self.structured_attribute_extraction_prompt.format(raw_attributes = extracted_raw_attributes, all_new_attributes=all_new_attributes)
            # extracted_structured_attributes = await self.text2column_text_model.run_inference(data_stream = structured_attribute_extraction_prompt)
            parsed_tuple_new_attributes = await self.process_new_attributes_tuple(extracted_structured_attributes = extracted_raw_attributes, new_attributes=new_attributes)
            batch_rows.append((current_attributes['video_id'], current_attributes['frame_id'], current_attributes['object_id'], parsed_tuple_new_attributes))

            if len(batch_rows) > Config.batch_size:
                yield batch_rows
                batch_rows = []
        
        yield batch_rows