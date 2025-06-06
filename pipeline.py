from pipelines.frame_extraction import VideoProcessor
from pipelines.captioning_embedding.captioning_pipeline import CaptioningPipeline
from pipelines.text2sql.text2sql_pipeline import Text2SQLPipeline
from pipelines.text2table.text2table_pipeline import Text2TablePipeline
from pipelines.text2column.text2column_pipeline import Text2ColumnPipeline
from database_integration import SQLLiteDBInterface, VectorDBInterface
from config.config import Config
from typing import Optional
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

        #extract raw captions for specific attributes and add columns to existing tables
        self.text2column_pipeline = Text2ColumnPipeline()

    async def generate_captions(self, video_path:str, video_filename:str):
        assert os.path.exists(os.path.join(video_path, video_filename)), f"ERROR: video filename {os.path.join(video_path, video_filename)} does not exist"
        async for sql_batch, vector_batch in self.video_processor.process_single_video(
            video_path=os.path.join(video_path, video_filename),
            video_id=video_filename,
            captioning_pipeline=self.captioning_pipeline,
            curr_vec_idx=-1
        ):
            self.sql_dbs.insert_many_rows_list(table_name = Config.caption_table_name, rows_data = sql_batch)
        self.text2table_pipeline.update_objects(self.captioning_pipeline.object_set)
    
    async def run_text2table(self, new_structured_table_name: Optional[str] = None, reboot: bool=False):
        if not reboot:
            #extract a combined caption from the raw table and create new tables from the schema the LLM generates
            total_num_rows = self.sql_dbs.get_total_num_rows(table_name=Config.caption_table_name)
            combined_description = self.sql_dbs.extract_concatenated_captions(table_name=Config.caption_table_name, attribute = 'description', num_rows=total_num_rows)
            structured_table_schemas = await self.text2table_pipeline.extract_table_schemas(all_captions = combined_description)
            self.sql_dbs.execute_script(structured_table_schemas)
        
        #extract and iterate all rows of the SQL db
        db_rows = self.sql_dbs.extract_all_rows(table_name = Config.caption_table_name)
        db_schema = self.sql_dbs.get_all_schemas_except_raw_videos() if not reboot else self.sql_dbs.get_table_schema(table_name=new_structured_table_name)
        
        obj_iterator = self.text2table_pipeline.run_pipeline_video(video_data=db_rows, database_schema=db_schema, reboot=reboot)
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

    async def insert_single_video(self, video_path:str, video_filename:str):
        await self.generate_captions(video_path = video_path, video_filename = video_filename)
        await self.run_text2table()
        
        #clear cached data in pipeline for multiple videos
        self.captioning_pipeline.clear_pipeline()
        self.text2sql_pipeline.clear_pipeline()
        self.text2table_pipeline.clear_pipeline()

    async def run_text2column(self, video_id: int, table_name: str, frame_batch: list, new_attributes_for_table: list):
        
        for attribute in new_attributes_for_table:
            self.sql_dbs.create_column(table_name=table_name, col_name=attribute, col_type=Config.new_col_type)

        table_cols = self.sql_dbs.get_table_schema(table_name=table_name, process=False)[:-1*len(new_attributes_for_table)]
        
        for (frame_id, frame) in frame_batch:
            table_rows = self.sql_dbs.execute_query(query=Config.object_detail_extraction_query.format(table_name = table_name, video_id=video_id, frame_id=frame_id))
            table_rows = [row[:-1*len(new_attributes_for_table)] for row in table_rows]
            async for sql_batch in self.text2column_pipeline.generate_new_column_data(table_name = table_name, frame=frame, table_rows=table_rows, table_columns=table_cols, new_attributes=list(new_attributes_for_table)):
                #function within text2col: regenerate raw caption for these frames (given row context and attribute data) + format into new columns
                self.sql_dbs.insert_rows_for_new_cols(table_name=table_name, col_names=list(new_attributes_for_table), data=sql_batch)


    async def process_query(self, language_query: str, llm_judge: bool):
        #extract the schema for the processed object table
        pdb.set_trace()
        table_schemas = self.sql_dbs.get_all_schemas_except_raw_videos()
        
        is_sufficient, sql_query, existing_tables_attributes_dict, new_tables_attributes_dict = await self.text2sql_pipeline.run_pipeline(
            question = language_query, 
            table_schemas = table_schemas, 
            llm_judge = llm_judge
        )
        print(f"is_sufficient: {is_sufficient}")
        print(f"sql_query: {sql_query}")
        print(f"existing_tables_attributes_dict: {existing_tables_attributes_dict}")
        print(f"new_tables_attributes_dict: {new_tables_attributes_dict}")
        pdb.set_trace()
        #only reboot with Text2Column if is_sufficient==False and existing_tables_attributes_dict has content
        if Config.text2column_enabled:
            if not is_sufficient and existing_tables_attributes_dict:
                #create new columns for existing tables
                for table_name, new_attributes in existing_tables_attributes_dict.items():
                    
                    unique_video_frame_ids = self.sql_dbs.get_unique_video_and_frame_ids(table_name=table_name, combined=True)

                    for video_id in unique_video_frame_ids:
                        async for frame_batch in self.video_processor.return_targeted_frames(video_path=os.path.join(Config.video_path, video_id), video_id=video_id, specific_frames=unique_video_frame_ids[video_id]):
                            #run text2column pipeline
                            await self.run_text2column(video_id = video_id, table_name=table_name, frame_batch = frame_batch, new_attributes_for_table=new_attributes)

            elif not is_sufficient and existing_tables_attributes_dict is None:
                raise RuntimeError("Error: cannot parse the query or cannot extract attributes")
        pdb.set_trace()
        #only reboot with NewTable if is_sufficient==False and new_tables_attributes_dict has content
        if Config.table_reboot_enabled:
            if not is_sufficient and new_tables_attributes_dict:

                (unique_video_ids, unique_frame_ids) = self.sql_dbs.get_unique_video_and_frame_ids(table_name=Config.caption_table_name)
                
                for video_id in unique_video_ids:
                    async for sql_batch, __ in self.video_processor.process_single_video(video_path=os.path.join(Config.video_path, video_id), video_id=video_id, captioning_pipeline=self.captioning_pipeline, curr_vec_idx=-1, new_attributes_dict=new_tables_attributes_dict, specific_frames=unique_frame_ids, reboot=True):
                        self.sql_dbs.insert_column_data(table_name=Config.caption_table_name, col_name=Config.temp_col_name, col_type=Config.temp_col_type, data=sql_batch)

                for new_table_name in new_tables_attributes_dict.keys():
                    table_schema = {key: "TEXT" for key in new_tables_attributes_dict[new_table_name]}
                    self.sql_dbs.add_new_table(table_name=new_table_name, table_schema=table_schema, table_prim_key=Config.processed_table_pk)
                    await self.run_text2table(new_structured_table_name=new_table_name, reboot=True)

            elif not is_sufficient and new_tables_attributes_dict is None:
                raise RuntimeError("Error: cannot parse the query or cannot extract attributes")
        #check query after rebooting once
        if not is_sufficient:
            table_schemas = self.sql_dbs.get_all_schemas_except_raw_videos()
            is_sufficient, sql_query, existing_tables_attributes_dict, new_tables_attributes_dict = await self.text2sql_pipeline.run_pipeline(
                question = language_query, 
                table_schemas = table_schemas, 
                llm_judge = llm_judge
            )
            print(f"is_sufficient: {is_sufficient}")
            print(f"sql_query: {sql_query}")
            print(f"existing_tables_attributes_dict: {existing_tables_attributes_dict}")
            print(f"new_tables_attributes_dict: {new_tables_attributes_dict}")
        if is_sufficient:
            result = self.sql_dbs.execute_query(query = sql_query)
            return result
        else:
            raise RuntimeError(f"Error in process_query: failed to process query {language_query}")

    async def process_many_queries(self, language_queries: list, llm_judge: bool):
        for query in language_queries:
            await self.process_query(language_query = query, llm_judge = llm_judge)
    
    async def insert_all_videos(self, video_path: str):
        # List all files in the directory
        all_files = os.listdir(video_path)
        
        # Filter to include only video files (e.g., .mp4, .mov)
        video_files = [f for f in all_files if f.endswith(('.mp4', '.mov'))]

        # Process each video file
        for video_filename in video_files:
            print(f"Processing video: {video_filename}")
            await self.insert_single_video(video_path=video_path, video_filename=video_filename)

if __name__ == '__main__':
    import time

    #PART 1: INSERTING/PROCESSING A VIDEO 
    query_pipeline = VideoQueryPipeline()
    # start_time = time.time()
    # asyncio.run(query_pipeline.insert_single_video(video_path='datasets/bdd', video_filename='00afa5b2-c14a542f.mov'))
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time}")
    
    #PART 2: SIMPLE QUERY FOR VIDEO
    # question = "In how many frames does a Chevrolet appear in front of a red light?"
    # question = "How many taxis are in the video?"
    # start_time = time.time()
    # result = asyncio.run(query_pipeline.process_query(language_query = question, llm_judge=Config.llm_judge))
    # end_time = time.time()
    # print("SYSTEM RESPONSE: ", result)
    # print(f"Time taken: {end_time - start_time}")

    #PART 3: MISSING TABLE QUERY FOR VIDEO
    # question = "When does the weather first have overcast after the first 5 frames?"
    question = "What is the first frame in which a damaged SUV stops at a red light?"
    start_time = time.time()
    result = asyncio.run(query_pipeline.process_query(language_query = question, llm_judge=Config.llm_judge))
    end_time = time.time()
    print("SYSTEM RESPONSE: ", result)
    




    # asyncio.run(query_pipeline.insert_single_video(video_path=Config.video_path, video_filename=Config.video_filename))
    # test_questions = [
    #     "What frames have the cabinet in it?",
    # ]
    # for question in test_questions:
    #     result = asyncio.run(query_pipeline.process_query(language_query = question, llm_judge=Config.llm_judge))
    #     print(f"Question: {question}")
    #     print(f"Result: {result}")
    #     print("------------------------")
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time}")