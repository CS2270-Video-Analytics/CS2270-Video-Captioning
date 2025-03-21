import sqlite3
import ollama
import json
import requests

class ElementExtractor:
    def __init__(self):
        self.model_name = 'llama2'  # Adjust based on your model
        assert self.is_ollama_model_running(), "ERROR: Ollama is not running. Run it on another terminal first"

    def is_ollama_model_running(self):
        try:
            response = requests.get('http://127.0.0.1:11434/api/version')
            return response.status_code == 200
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return False

    def create_extraction_prompt(self, caption: str) -> str:
        return f"""Extract structured information from this caption into objects, their attributes, and actions.
For each object mentioned, provide:
- object type (e.g., road, car, building)
- attributes (comma-separated list of descriptive features)
- actions (comma-separated list of actions, or None if no action)

Format the response as a list of JSON objects:
[
    {{"object": "road", "attributes": "two lanes, asphalt", "actions": "None"}},
    {{"object": "car", "attributes": "white, dark interior", "actions": "driving towards camera"}}
]

Caption:
{caption}

Only include objects with clear descriptions. If no attributes or actions, use "None".
"""

    def process_caption(self, caption: str):
        """Process a single caption and return structured data"""
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': self.create_extraction_prompt(caption)
            }]
        )
        return json.loads(response['message']['content'])

    def process_database(self, db_path: str = 'captions.db'):
        """Process all captions in the database and create a new structured table"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create new table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scene_elements (
            video_id TEXT,
            frame_id REAL,
            object_type TEXT,
            attributes TEXT,
            actions TEXT,
            PRIMARY KEY (video_id, frame_id, object_type)
        )
        """)

        # Get all captions
        cursor.execute("SELECT video_id, frame_id, caption FROM captions")
        rows = cursor.fetchall()

        for video_id, frame_id, caption in rows:
            print(f"Processing video {video_id}, frame {frame_id}")
            elements = self.process_caption(caption)

            for element in elements:
                cursor.execute("""
                INSERT OR REPLACE INTO scene_elements 
                (video_id, frame_id, object_type, attributes, actions)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    video_id,
                    frame_id,
                    element['object'],
                    element['attributes'],
                    element['actions']
                ))

        conn.commit()
        conn.close()

def main():
    extractor = ElementExtractor()
    extractor.process_database()

if __name__ == "__main__":
    main()