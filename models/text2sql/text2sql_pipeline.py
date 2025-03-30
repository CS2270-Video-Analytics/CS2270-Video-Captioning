import sqlite3
import sqlparse
from config.config import Config
if Config.debug:
    import pdb

class Text2SQLPipeline():

    def __init__(self, text2sql_func):
        """
        Initializes the Text2SQLPipeline with a text-to-SQL function.

        Parameters:
            text2sql_func (function): A function that generates SQL queries given a question and schema.
        """
        self.model = text2sql_func

    def run_pipeline(self, question:str, db_file):
        """
        Converts a natural language question into an SQL query and executes it on a given SQLite database.

        Parameters:
            question (str): The natural language question to convert to SQL.
            db_file (str): Path to the SQLite database file.
            text2sql_func (function): A function that generates SQL queries given a question and schema.

        Returns:
            tuple: The generated SQL query and its execution result.
        """
        db_schema = self.get_schema(db_file, ["object_detections"])
        print(db_schema)
        sql_query = self.normalize_sql(self.model(question, db_schema))
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

    def get_schema(self, db_file, tables_to_include):
        """
        Extracts the schema information from specific tables in an SQLite database.

        Parameters:
            db_file (str): Path to the SQLite database file.
            tables_to_include (list of str): List of table names to include in the schema.

        Returns:
            str: The schema of specified tables, including column details.
        """
        import sqlite3

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        try:
            # Verify table names against sqlite_master
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            all_tables = {row[0] for row in cursor.fetchall()}

            # Filter only existing tables
            valid_tables = [t for t in tables_to_include if t in all_tables]
            if not valid_tables:
                return "None of the specified tables exist in the database."

            schema_info = []

            for table_name in valid_tables:
                schema_info.append(f"Table: {table_name}")

                # Get column details
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()

                for col in columns:
                    col_id, col_name, col_type, _, _, _ = col
                    schema_info.append(f"  - {col_name} ({col_type})")

                schema_info.append("\n")  # Add space between tables

            conn.close()
            return "\n".join(schema_info)

        except Exception as e:
            conn.close()
            return f"Error retrieving schema: {str(e)}"

    def normalize_sql(self, sql):
        """Remove extraneous formatting, code block markers, and standardize SQL."""
        sql = sql.strip().replace("```sql", "").replace("```", "").strip()  # Remove markdown SQL blocks
        return sqlparse.format(sql, reindent=True, keyword_case='upper').strip()  # Standardize format