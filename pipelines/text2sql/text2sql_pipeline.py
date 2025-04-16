import sqlite3
import sqlparse
import logging
from config.config import Config
if Config.debug:
    import pdb
from models.text2sql import Text2SQLModelFactory
from models.language_models.OpenAIText import OpenAIText
# Set up logging
logger = logging.getLogger(__name__)

class Text2SQLPipeline():
    """Pipeline for converting natural language questions to SQL queries and executing them"""

    def __init__(self):
        """
        Initialize the Text2SQL pipeline with the model specified in the config
        """
        try:
            # Initialize the OpenAIText model with parameters from config
            self.model = OpenAIText(
                model_params=Config.text2sql_params,
                model_name=Config.text2sql_model_name.split(';')[1],
                model_precision=Config.model_precision,
                system_eval=Config.system_eval
            )
            logger.info(f"Initialized Text2SQL pipeline with OpenAIText model: {self.model.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Text2SQL pipeline: {str(e)}")
            raise

    def run_pipeline(self, question: str, db_file: str):
        """
        Converts a natural language question into an SQL query using the existing database schema.

        Parameters:
            question (str): The natural language question to convert to SQL.
            db_file (str): Path to the SQLite database file.

        Returns:
            str: The generated SQL query.
        """
        try:
            table_schema = self.get_existing_schema(db_file)
            prompt = Config.get_text2sql_prompt(table_schema, question)
            sql_query, info = self.model.run_inference(prompt)
            if info['error']:
                raise Exception(info['error'])
            logger.debug(f"Generated SQL query: {sql_query}")
            return sql_query
        except Exception as e:
            logger.error(f"Error in run_pipeline: {str(e)}")
            raise
    
    def clear_pipeline(self):
        #clear the cache that remains for previous runs of text2table
        self.table_attributes = []
        self.all_objects = []

    def execute_sql(self, db_file, sql_query):
        """
        Executes an SQL query on a given SQLite database.

        Parameters:
            db_file (str): Path to the SQLite database file.
            sql_query (str): The SQL query to execute.

        Returns:
            list: Query result as a list of tuples, or an error message if execution fails.
        """
        # Connect to the SQLite database
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            cursor.execute(sql_query)
            result = cursor.fetchall()  # Fetch query results
            logger.debug(f"SQL execution result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            if conn:
                conn.close()

    def normalize_sql(self, sql):
        """
        Remove extraneous formatting, code block markers, and standardize SQL.
        
        Parameters:
            sql (str): The SQL query to normalize
            
        Returns:
            str: The normalized SQL query
        """
        sql = sql.strip().replace("```sql", "").replace("```", "").strip()  # Remove markdown SQL blocks
        return sqlparse.format(sql, reindent=True, keyword_case='upper').strip()  # Standardize format

    def get_existing_schema(self, db_file):
        """
        Retrieve the schema of existing tables in the database.

        Parameters:
            db_file (str): Path to the SQLite database file.

        Returns:
            dict: A dictionary with table names as keys and column information as values.
        """
        schema = {}
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                schema[table_name] = [(col[1], col[2]) for col in columns]  # (column_name, data_type)
        except Exception as e:
            logger.error(f"Error retrieving schema: {str(e)}")
        finally:
            if conn:
                conn.close()
        return schema

if __name__ == "__main__":
    pipeline = Text2SQLPipeline()
    test_questions = [
        "Find all video_id and frame_id where someone is standing",
        "Find all video_id and frame_id where someone is in the kitchen"
    ]
    
    print("\n=== Testing Text2SQL Pipeline ===")
    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            sql_query = pipeline.run_pipeline(question, Config.db_path)
            print(f"Generated SQL: {sql_query}")
            results = pipeline.execute_sql(Config.db_path, sql_query)
            print(results)
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)