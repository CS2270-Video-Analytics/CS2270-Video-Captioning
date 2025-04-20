from pipelines.frame_extraction import VideoProcessor
from pipelines.captioning_embedding.captioning_pipeline import CaptioningPipeline
from pipelines.text2sql.text2sql_pipeline import Text2SQLPipeline
from pipelines.text2table.text2table_pipeline import Text2TablePipeline
from database_integration import SQLLiteDBInterface, VectorDBInterface
from config.config import Config
import os
import asyncio
import pdb
import torch

class VideoQueryPipeline():
    def __init__(self):
        #video sampling module to sample videos
        self.video_processor = VideoProcessor()

        #create pipeline component for captioning
        self.captioning_pipeline = CaptioningPipeline()

        #SQL db containing raw and processed SQL table
        self.sql_dbs = SQLLiteDBInterface(Config.sql_db_name)

        #vector db containing image embeddings
        # Since we are no longer doing VectorDB, we can comment this out
        # self.vector_db = VectorDBInterface(vector_dim = self.captioning_pipeline.clip_model.clip_model.visual.output_dim)

        #natural language 2 sql generation pipeline
        self.text2sql_pipeline = Text2SQLPipeline()

        #raw caption 2 formatted table pipeline
        self.text2table_pipeline = Text2TablePipeline(all_objects = [], db_path=Config.db_path)

    async def generate_captions(self, video_path:str, video_filename:str):
        assert os.path.exists(os.path.join(video_path, video_filename)), f"ERROR: video filename {video_filename} does not exist"
        async for sql_batch, vector_batch in self.video_processor.process_single_video(
            video_path=os.path.join(video_path, video_filename),
            video_id=video_filename,
            captioning_pipeline=self.captioning_pipeline,
            curr_vec_idx=-1
        ):
            self.sql_dbs.insert_many_rows_list(table_name = Config.caption_table_name, rows_data = sql_batch)
        self.text2table_pipeline.update_objects(self.captioning_pipeline.object_set)
    
    async def run_text2table(self):
        #extract a combined caption from the raw table and create new tables from the schema the LLM generates
        total_num_rows = self.sql_dbs.get_total_num_rows(table_name=Config.caption_table_name)
        combined_description = self.sql_dbs.extract_concatenated_captions(table_name=Config.caption_table_name, attribute = 'description', num_rows=total_num_rows)
        structured_table_schemas = await self.text2table_pipeline.extract_table_schemas(all_captions = combined_description)
        self.sql_dbs.execute_script(structured_table_schemas)
        
        #extract and iterate all rows of the SQL db
        db_rows = self.sql_dbs.extract_all_rows(table_name = Config.caption_table_name)
        db_schema = self.sql_dbs.get_all_schemas_except_raw_videos()
        obj_iterator = self.text2table_pipeline.run_pipeline_video(video_data=db_rows, database_schema=db_schema)
        #insert a batch of rows into the SQL object db
        batch_count = 0
        row_count = 0
        # Use async for to iterate over an iterator produced by an async function
        async for data_batch in obj_iterator:
            batch_count += 1
            for frame_caption in data_batch:
                # Dictionary {table: a list of rows} where each row is object level
                frame_caption_dict = frame_caption[0]
                for table_name, rows_data in frame_caption_dict.items():
                    self.sql_dbs.insert_many_rows_list(table_name=table_name, rows_data=rows_data)
                    row_count += len(rows_data)
            print(f"[Progress] Processed batch {batch_count} — total rows inserted: {row_count}")
        print(f"[Done] All {batch_count} batches processed. Total rows inserted: {row_count}")

    async def process_single_video(self, video_path:str, video_filename:str):
        await self.generate_captions(video_path = video_path, video_filename = video_filename)
        await self.run_text2table()
        
        #clear cached data in pipeline for multiple videos
        self.captioning_pipeline.clear_pipeline()
        self.text2sql_pipeline.clear_pipeline()
        self.text2table_pipeline.clear_pipeline()
    
    async def process_query(self, language_query: str, llm_judge: bool):
        #extract the schema for the processed object table
        table_schemas = self.sql_dbs.get_all_schemas_except_raw_videos()
        
        #parse the language query into a SQL query
        is_sufficient, sql_query, existing_tables_attributes_dict, new_tables_attributes_dict = await self.text2sql_pipeline.run_pipeline(
            question = language_query, 
            table_schemas = table_schemas, 
            llm_judge = llm_judge
        )
        print(f"is_sufficient: {is_sufficient}")
        print(f"sql_query: {sql_query}")
        print(f"existing_tables_attributes_dict: {existing_tables_attributes_dict}")
        print(f"new_tables_attributes_dict: {new_tables_attributes_dict}")

        #execute query on the sql db
        self.sql_dbs.execute_query(query = sql_query)

    async def process_all_videos(self, video_path: str):
        # List all files in the directory
        all_files = os.listdir(video_path)
        
        # Filter to include only video files (e.g., .mp4, .mov)
        video_files = [f for f in all_files if f.endswith(('.mp4', '.mov'))]

        # Process each video file
        for video_filename in video_files:
            print(f"Processing video: {video_filename}")
            await self.process_single_video(video_path=video_path, video_filename=video_filename)

if __name__ == '__main__':
    query_pipeline = VideoQueryPipeline()
    pdb.set_trace()
    # asyncio.run(query_pipeline.process_single_video(video_path=Config.video_path, video_filename=Config.video_filename))
    test_questions = [
        "What frames have the cabinet in it?",
        "Is there a cabinet in the video?",
        "What color is the cabinet?"
    ]
    for question in test_questions:
        asyncio.run(query_pipeline.process_query(language_query = question, llm_judge=Config.llm_judge))
