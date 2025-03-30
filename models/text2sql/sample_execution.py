import os

# from text2sql_pipeline import text2sql_pipeline
from models.text2sql.text2sql_models import create_text2sql_func_openai, create_text2sql_func_deepseek, create_text2sql_func_anthropic
from models.text2sql.text2sql_hf import create_text2sql_func_hf
from models.text2sql.text2sql_pipeline import Text2SQLPipeline
from models.text2sql.run_text2table import create_table, populate_table

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPIDER_PATH = os.path.join(BASE_DIR, "datasets", "Spider_test")
DB_PATH = os.path.join(SPIDER_PATH, "test_database")  # SQLite databases

db_id = "soccer_3"
db_file = os.path.join(DB_PATH, db_id, f"{db_id}.sqlite")

db_file = os.path.join(SPIDER_PATH, "video_frames.db")

output_db_file = os.path.join(SPIDER_PATH, "output_video_frames.db")

populate_table(db_file, output_db_file)

# create_table(db_file, output_db_file)

# # question = "What are the earnings of players from either of the countries of Australia or Zimbabwe?"
# question = "Do we have a black van?"
# # question = "How many clubs are there?"
# # text2sql_func = create_text2sql_func_openai("gpt-3.5-turbo")
# text2sql_func = create_text2sql_func_openai("gpt-4o", categories=["Vehicle", "Person"])
# # text2sql_func = create_text2sql_func_hf("apple/OpenELM-270M")
# # text2sql_func = create_text2sql_func_deepseek()
# # text2sql_func = create_text2sql_func_anthropic()
# # text2sql_func = create_text2sql_func_hf("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# # text2sql_func = create_text2sql_func_hf("gaussalgo/T5-LM-Large-text2sql-spider")

# pipeline = Text2SQLPipeline(text2sql_func)
# sql_query = pipeline.run_pipeline(question, db_file)
# print(f"SQL Query: {sql_query}")
# # Execute the SQL query
# result = pipeline.execute_sql(db_file, sql_query)
# print(f"Execution Result: {result}")
