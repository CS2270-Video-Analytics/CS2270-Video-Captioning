import os
import pytest
from database_integration.sqllite_db_interface import SQLLiteDBInterface
from config import Config

@pytest.fixture(scope="module")
def db_interface():
    # Set up a test database
    test_db_name = "test_video_frames.db"
    test_caption_table_name = "raw_videos"
    test_caption_table_schema = {
        'video_id': "TEXT NOT NULL", 
        'frame_id': "REAL NOT NULL", 
        'description': "TEXT NOT NULL", 
        'vector_id': "INTEGER"
    }
    test_caption_table_pk = ['video_id', 'frame_id']
    test_table_name_schema_dict = {
        test_caption_table_name: [test_caption_table_schema, test_caption_table_pk]
    }
    db_interface = SQLLiteDBInterface(db_name=test_db_name, table_name_schema_dict=test_table_name_schema_dict)
    yield db_interface
    # Teardown: Remove the test database after tests
    db_interface.close_conn()
    os.remove(os.path.join(Config.sql_db_path, test_db_name))

def test_create_table(db_interface):
    # Test if the table is created correctly
    schema = db_interface.get_table_schema("raw_videos")
    assert "raw_videos" in schema

def test_insert_and_retrieve(db_interface):
    # Test inserting and retrieving data
    test_data = [('cd31bcc0-07b8e93f.mov', '2.0000832639467108', 'centered in the frame', 1)]
    db_interface.insert_many_rows_list("raw_videos", test_data)
    rows = db_interface.extract_all_rows("raw_videos")
    assert len(rows) == 1
    assert rows[0][0] == "cd31bcc0-07b8e93f.mov"

def test_primary_key(db_interface):
    schema_info = db_interface.extract_schema_dict()
    primary_keys = schema_info["raw_videos"][1]
    assert "video_id" in primary_keys
    assert "frame_id" in primary_keys