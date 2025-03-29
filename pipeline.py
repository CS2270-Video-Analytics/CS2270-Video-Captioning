from data_processing import VideoProcessor
from database_integration.sqlite_db_interface import SQLLiteDBInterface
from database_integration.vector_db_interface import VectorDBInteface
from models.captioning_embedding.captioning_pipeline import CaptioningPipeline
# from models.text2sql
from models.text2table.text2table_pipeline import Text2TablePipeline


class VideoQueryPipeline():

    def __init__(self):
