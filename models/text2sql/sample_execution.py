import os

from text2sql_pipeline import text2sql_pipeline
from text2sql_openai import create_text2sql_func_openai

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPIDER_PATH = os.path.join(BASE_DIR, "datasets", "Spider_test")
DB_PATH = os.path.join(SPIDER_PATH, "test_database")  # SQLite databases

db_id = "soccer_3"
db_file = os.path.join(DB_PATH, db_id, f"{db_id}.sqlite")

question = "What are the earnings of players from either of the countries of Australia or Zimbabwe?"
text2sql_func = create_text2sql_func_openai("gpt-3.5-turbo")

print(text2sql_pipeline(question, db_file, text2sql_func))