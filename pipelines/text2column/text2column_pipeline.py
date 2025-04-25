from config.config import Config
from typing import List, Dict
from models.language_models.Anthropic import Anthropic
from models.language_models.OpenAIText import OpenAIText
from models.language_models.OllamaText import OllamaText
from models.language_models.DeepSeek import DeepSeek
import sqlite3
import asyncio
import pdb

class Text2ColumnPipeline():
    def __init__(self, all_objects:List[str], db_path: str):
        self.text2table_frame_prompt = Config.text2table_frame_prompt
        # self.text2table_frame_context = Config.text2table_frame_context

        #store all prompts for text2table
        self.attribute_extraction_prompt = Config.text2table_attribute_extraction_prompt
        self.schema_extraction_prompt_format = Config.text2table_schema_generation_prompt

        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.table_context = self.update_table_context()

        #initialize the model that needs to be used for captioning
        model_options = {'Ollama': OllamaText, 'OpenAI': OpenAIText, 'Anthropic': Anthropic, 'DeepSeek':DeepSeek}
        [text2colmn_model, text2colmn_model_name] = Config.caption_model_name.split(';')
        assert text2colmn_model in model_options, f'ERROR: model {text2colmn_model} does not exist or is not supported yet'
        
        self.text2column_vision_model = model_options[text2colmn_model](
                                    model_params = Config.text2column_params, 
                                    model_name=text2colmn_model_name, 
                                    model_precision=Config.model_precision, 
                                    system_eval=Config.system_eval)
        self.text2column_text_model = model_options[text2colmn_model](
                                    model_params = Config.text2column_params, 
                                    model_name=text2colmn_model_name, 
                                    model_precision=Config.model_precision, 
                                    system_eval=Config.system_eval)
    
    def process_new_attributes_tuple(self, extracted_structured_attributes: str):
        return extracted_structured_attributes

    async def generate_new_column_data(self, frame, table_rows: list, table_columns: list, new_attributes: set):
        batch_rows= []
        for row in table_rows:
            all_new_attributes = ','.join(new_attributes)
            current_attributes = {k:v for (k,v) in zip(table_columns, row)}

            attribute_extraction_prompt = self.attribute_extraction_prompt.format(current_attributes = str(current_attributes), all_new_attributes=all_new_attributes)
            extracted_raw_attributes = await self.text2column_vision_model.run_inference(frame, dict(prompt = attribute_extraction_prompt))
            extracted_structured_attributes = await self.text2column_text_model.run_inference(data_stream = extracted_raw_attributes)
            parsed_tuple_new_attributes = await self.process_new_attributes_tuple(extracted_structured_attributes = extracted_structured_attributes)
            batch_rows.append(parsed_tuple_new_attributes)

            if len(batch_rows) > Config.batch_size:
                yield batch_rows
                batch_rows = []
            
    def build_json_template(self, schema_dict, frame_id_placeholder="{frame_id}", video_id_placeholder="{video_id}"):
        lines = []
        for table, cols in schema_dict.items():
            lines.append(f'  "{table}": [')
            lines.append("    {")
            for col in cols:
                if col == "frame_id":
                    value = f'"{frame_id_placeholder}"'
                elif col == "video_id":
                    value = f'"{video_id_placeholder}"'
                else:
                    value = '...'
                lines.append(f'      "{col}": {value},')
            lines[-1] = lines[-1].rstrip(",")  # remove trailing comma
            lines.append("    }")
            lines.append("  ],")
        lines[-1] = lines[-1].rstrip(",")  # remove trailing comma
        return "\n".join(lines)
    
