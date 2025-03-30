from data_processing.process_videos import VideoProcessor
from models.captioning_embedding.captioning_pipeline import CaptioningPipeline
from models.text2sql.text2sql_pipeline import Text2SQLPipeline
from models.text2table.text2table_pipeline import Text2TablePipeline
from database_integration import SQLLiteDBInterface, VectorDBInterface
from config.config import Config
import os
import pdb
import torch

class VideoQueryPipeline():

    def __init__(self):
        #video sampling module to sample videos
        self.video_processor = VideoProcessor()

        #create pipeline component for captioning
        self.captioning_pipeline = CaptioningPipeline()

        #SQL db containing raw and processed SQL table
        self.sql_dbs = SQLLiteDBInterface()

        #vector db containing image embeddings
        self.vector_db = VectorDBInterface(vector_dim = self.captioning_pipeline.clip_model.visual.output_dim)

        #natural language 2 sql generation pipeline
        #TODO: debug this afterwards
        # self.text2sql_pipeline = Text2SQLPipeline()

        #raw caption 2 formatted table pipeline
        self.text2table_pipeline = Text2TablePipeline(all_objects = [])


    def process_video(self, video_path:str, video_filename:str):

        #first check for valid video path
        assert os.path.exists(os.path.join(video_path, video_filename)), f"ERROR: video filename {video_filename} does not exist"

        #process the video to get a frame extractor
        frame_iterator = self.video_processor.process_single_video(video_path = os.path.join(video_path, video_filename), video_id=video_filename, captioning_pipeline = self.captioning_pipeline, curr_vec_idx = self.vector_db.get_num_vecs())

        #iterate all frame batches and add to both sql and vector DBs
        while True:
            try:
                sql_batch, vector_batch = next(frame_iterator)

                #insert rows into the SQL db
                self.sql_dbs.insert_many_rows_list(table_name = Config.caption_table_name, rows_data = sql_batch)

                #insert vectors into the vector db
                vector_batch = torch.cat(vector_batch, dim=0)
                self.vector_db.insert_many_vectors(vectors = vector_batch)
                
            except StopIteration:
                break

        
        #once all batches of frames (vectors and raw captions) have been added, start text to table pipeline
        self.text2table_pipeline.update_objects(self.captioning_pipeline.object_set) #first update with list of all objects found in the video
        #save the vector db after processing
        self.vector_db.save_vectordb()

        #extract and iterate all rows of the SQL db
        db_rows = self.sql_dbs.extract_all_rows(table_name = Config.caption_table_name)
        obj_iterator = self.text2table_pipeline.run_pipeline_video(db_rows)

        #insert a batch of rows into the SQL object db
        while True:
            try:
                data_batch = next(obj_iterator)
                self.sql_dbs.insert_many_rows_list(table_name = Config.processed_table_name, rows_data = data_batch)
            except StopIteration:
                break


    
    def process_query(self, language_query:str):

        #extract the schema for the processed object table
        table_schema = self.sql_dbs.get_schema(table_name = [Config.processed_table_name])

        #parse the language query into a SQL query
        sql_query = self.text2sql_pipeline.run_pipeline(question = language_query, db_schema = table_schema)


if __name__ == '__main__':
    pdb.set_trace()
    dummy = VideoQueryPipeline()

    video_path = '/users/ssunda11/git/CS2270-Video-Captioning/datasets/BDD_test'
    filename = 'test2.mov'

    dummy.process_video(video_path = video_path, video_filename = filename)
        














