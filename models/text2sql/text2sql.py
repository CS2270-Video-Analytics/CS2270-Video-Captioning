import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import sqlite3
import json
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPIDER_PATH = os.path.join(BASE_DIR, "datasets", "Spider_test")
DB_PATH = os.path.join(SPIDER_PATH, "test_database")  # SQLite databases
QUESTIONS_FILE = os.path.join(SPIDER_PATH, "test.json")  # Test questions
GOLD_SQL_FILE = os.path.join(SPIDER_PATH, "test_gold.sql")  # Ground truth SQL

# Load test questions
with open(QUESTIONS_FILE, "r") as f:
    test_data = json.load(f)

# Load ground truth SQL queries (test_gold.sql)
with open(GOLD_SQL_FILE, "r") as f:
    gold_sql_lines = f.readlines()

# Create a mapping of db_id to ground truth SQL queries
gold_sql_mapping = {}
for line in gold_sql_lines:
    parts = line.strip().split("\t")  # Assuming tab-separated values
    if len(parts) == 2:
        db_id, sql_query = parts
        gold_sql_mapping[db_id] = sql_query

_ = load_dotenv(find_dotenv()) # read local .env file
OpenAI.api_key = os.environ['OPENAI_API_KEY']

llm_model = "gpt-3.5-turbo"

client = OpenAI()

def generate_sql(question, schema_info, llm_model):
    """Generate SQL query from a natural language question using OpenAI API."""
    prompt = f"""
    You are an AI that converts natural language questions into SQL queries.
    Use the provided database schema to generate the correct SQL query.

    Database Schema:
    {schema_info}

    Question: "{question}"
    SQL Query:
    """
    messages=[{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def execute_sql(db_id, sql_query):
    """Execute the generated SQL query on the corresponding SQLite database."""
    db_file = os.path.join(DB_PATH, db_id, f"{db_id}.sqlite")

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        cursor.execute(sql_query)
        result = cursor.fetchall()  # Fetch query results
        conn.close()
        return result
    except Exception as e:
        conn.close()
        return f"Error: {str(e)}"

def get_schema_from_sqlite(db_id):
    """Extract schema information directly from the SQLite database."""
    db_file = os.path.join(DB_PATH, db_id, f"{db_id}.sqlite")

    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            return "No tables found in database."

        schema_info = []

        for table_name in tables:
            table_name = table_name[0]
            schema_info.append(f"Table: {table_name}")

            # Get column details
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            for col in columns:
                col_id, col_name, col_type, _, _, _ = col
                schema_info.append(f"  - {col_name} ({col_type})")

            schema_info.append("\n")  # Add space between tables

        conn.close()
        return "\n".join(schema_info)  # Return formatted schema

    except Exception as e:
        conn.close()
        return f"Error retrieving schema: {str(e)}"

db_id = "soccer_3"
question = "What are the earnings of players from either of the countries of Australia or Zimbabwe?"
db_schema = get_schema_from_sqlite(db_id)
generated_sql = generate_sql(question, db_schema, llm_model)
print(generated_sql)

sql_output = execute_sql(db_id, generated_sql)
print(sql_output)
# def get_completion(prompt, model=llm_model):
#     messages = [{"role": "user", "content": prompt}]
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#     )
#     return response.choices[0].message.content

# print(get_completion("What is 1+1?"))