import sys
import os
from config.config import Config

if Config.debug:
    import cv2
    import pdb
    from torchvision import transforms

from .OllamaText import OllamaText
import torch
import sqlite3


class Text2TablePipeline:

    def __init__(self):

        if Config.text2table_model == 'Seq2Seq':
            raise NotImplementedError("ERROR: not implemented yet")
        elif Config.text2table_model == 'GPT3':
            self.text2table_incontext_prompt = Config.text2table_prompt
        elif Config.text2table_model == 'Ollama':
            self.text2table_incontext_prompt = Config.text2table_prompt
        
        #store prompt for per frame information extraction
        self.text2table_frame_prompt = Config.text2table_frame_prompt
        
        #initialize the model that needs to be used for captioning
        model_options = {'Ollama': OllamaText}
        assert Config.text2table_model in model_options, f'ERROR: model {Config.text2table_model} does not exist or is not supported yet'
        self.text2table_model = model_options[Config.text2table_model]()

    
    def run_pipeline(self, caption: str, video_id:int, frame_id:int):

        #create the overall prompt structure
        text2table_frame_prompt = self.text2table_frame_prompt.format(caption=caption, video_id=video_id, frame_id=frame_id)

        #generate the structured caption using text2table model
        structured_caption, info = self.text2table_model.run_inference(query = text2table_frame_prompt if Config.text2table_model == 'Seq2Seq' else '\n'.join(self.text2table_incontext_prompt, text2table_frame_prompt))

        #parse the structured caption as a partitioned of structured elements in JSON
        db_data_row = self.parse_table_output_json(structured_caption, video_id, frame_id)

        return db_data_row
    
    #NOTE: to be reconfigured if and when we use actual Text2Table formatting with <s> and <t> separators
    def parse_table_output_json(self, structured_caption:str, video_id:int, frame_id:int):

        # Regular expression pattern to match dictionary-like structures
        pattern = r'\{\{(.*?)\}\}'

        # Find all matches
        parsed_rows = re.findall(pattern, structured_caption, re.DOTALL)

        # Convert extracted matches to valid Python dictionaries
        parsed_rows = [ast.literal_eval("{" + row.strip() + "}") for row in parsed_rows]
        
        for r in parsed_rows:
            r['video_id'] = video_id
            r['frame_id'] = frame_id
        

        return result
    
    def parse_table_output_t2t(self, structured_caption:str, video_id:int, frame_id:int):

        # Split by new-line token
        rows = t.split("<n>")

        # Parse each row by removing leading/trailing spaces and splitting on <s>
        parsed_rows = [list(filter(None, row.split("<s>"))) for row in rows]

        # Remove extra whitespace and commas
        parsed_rows = [[entry.strip(" ,") for entry in row] for row in parsed_rows]
        
        for r in parsed_rows:
            r['video_id'] = video_id
            r['frame_id'] = frame_id
        

        return result













# class ElementExtractor:
#     def __init__(self):
#         self.model_name = 'llama2'  # Adjust based on your model
#         assert self.is_ollama_model_running(), "ERROR: Ollama is not running. Run it on another terminal first"

#     def is_ollama_model_running(self):
#         try:
#             response = requests.get('http://127.0.0.1:11434/api/version')
#             return response.status_code == 200
#         except Exception as e:
#             print(f"Error connecting to Ollama: {e}")
#             return False

#     def create_extraction_prompt(self, caption: str) -> str:
#         return f"""Extract structured information from this caption into objects, their attributes, and actions.
# For each object mentioned, provide:
# - object type (e.g., road, car, building)
# - attributes (comma-separated list of descriptive features)
# - actions (comma-separated list of actions, or None if no action)

# Format the response as a list of JSON objects:
# [
#     {{"object": "road", "attributes": "two lanes, asphalt", "actions": "None"}},
#     {{"object": "car", "attributes": "white, dark interior", "actions": "driving towards camera"}}
# ]

# Caption:
# {caption}

# Only include objects with clear descriptions. If no attributes or actions, use "None".
# """

#     def process_caption(self, caption: str):
#         """Process a single caption and return structured data"""
#         response = ollama.chat(
#             model=self.model_name,
#             messages=[{
#                 'role': 'user',
#                 'content': self.create_extraction_prompt(caption)
#             }]
#         )
#         return json.loads(response['message']['content'])

#     def process_database(self, db_path: str = 'captions.db'):
#         """Process all captions in the database and create a new structured table"""
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         # Create new table if it doesn't exist
#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS scene_elements (
#             video_id TEXT,
#             frame_id REAL,
#             object_type TEXT,
#             attributes TEXT,
#             actions TEXT,
#             PRIMARY KEY (video_id, frame_id, object_type)
#         )
#         """)

#         # Get all captions
#         cursor.execute("SELECT video_id, frame_id, caption FROM captions")
#         rows = cursor.fetchall()

#         for video_id, frame_id, caption in rows:
#             print(f"Processing video {video_id}, frame {frame_id}")
#             elements = self.process_caption(caption)

#             for element in elements:
#                 cursor.execute("""
#                 INSERT OR REPLACE INTO scene_elements 
#                 (video_id, frame_id, object_type, attributes, actions)
#                 VALUES (?, ?, ?, ?, ?)
#                 """, (
#                     video_id,
#                     frame_id,
#                     element['object'],
#                     element['attributes'],
#                     element['actions']
#                 ))

#         conn.commit()
#         conn.close()

def main():
    extractor = ElementExtractor()
    extractor.process_database()

if __name__ == "__main__":
    main()