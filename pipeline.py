from data_processing.process_videos import VideoProcessor
from database_integration.sqlite_db_interface import SQLLiteDBInterface
from database_integration.vector_db_interface import VectorDBInteface
from models.captioning_embedding.captioning_pipeline import CaptioningPipeline
from models.text2sql.text2sql_pipeline import Text2SQLPipeline
from models.text2table.text2table_pipeline import Text2TablePipeline
from config.config import Config

class VideoQueryPipeline():

    def __init__(self):
        #video sampling module to sample videos
        self.video_processor = VideoProcessor()

        #create pipeline component for captioning
        self.captioning_pipeline = CaptioningPipeline()


        #SQL db containing raw and processed SQL table
        self.sql_dbs = SQLLiteDBInterface()

        #vector db containing image embeddings
        self.vector_db = VectorDBInteface()





