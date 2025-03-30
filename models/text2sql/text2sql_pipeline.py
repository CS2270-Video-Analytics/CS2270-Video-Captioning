import sqlite3
import sqlparse
from config.config import Config
if Config.debug:
    import pdb

class Text2SQLPipeline():

    def __init__(self):

        model_options = {'OpenAI': create_text2sql_func_openai, 'DeepSeek': create_text2sql_func_deepseek, 'Anthropic': create_text2sql_func_anthropic, 'HuggingFace': create_text2sql_func_hf}
        assert Config.text2sql_model in model_options, f'ERROR: model {Config.text2sql_model} does not exist or is not supported yet'
        self.model = model_options[Config.text2sql_model](model_name = Config.text2sql_model_name)

    def run_pipeline(self, question:str, table_schema:str):
        """
        Converts a natural language question into an SQL query and executes it on a given SQLite database.

        Parameters:
            question (str): The natural language question to convert to SQL.
            db_file (str): Path to the SQLite database file.
            text2sql_func (function): A function that generates SQL queries given a question and schema.

        Returns:
            tuple: The generated SQL query and its execution result.
        """
        sql_query = self.normalize_sql(self.model(question, table_schema))
        # sql_output = self.execute_sql(db_file, sql_query)
        return sql_query

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

    def normalize_sql(self, sql):
        """Remove extraneous formatting, code block markers, and standardize SQL."""
        sql = sql.strip().replace("```sql", "").replace("```", "").strip()  # Remove markdown SQL blocks
        return sqlparse.format(sql, reindent=True, keyword_case='upper').strip()  # Standardize format