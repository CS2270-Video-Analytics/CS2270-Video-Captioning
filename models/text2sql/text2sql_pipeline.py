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
        db_schema = self.get_schema(db_file, tables_to_include=None)
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

    def get_schema(self, db_file, tables_to_include=None):
        """
        Extracts the schema information from specified tables in an SQLite database.
        If no tables are specified, returns the schema of all tables.

        Parameters:
            db_file (str): Path to the SQLite database file.
            tables_to_include (list of str, optional): List of table names to include. If None, all tables are used.

        Returns:
            str: The schema of the selected tables, including column details.
        """
        import sqlite3

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        try:
            # Get all table names from sqlite_master
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            all_tables = {row[0] for row in cursor.fetchall()}

            # If no specific tables given, use all
            if not tables_to_include:
                valid_tables = list(all_tables)
            else:
                valid_tables = [t for t in tables_to_include if t in all_tables]

            if not valid_tables:
                return "None of the specified tables exist in the database."

            schema_info = []

            for table_name in sorted(valid_tables):
                schema_info.append(f"Table: {table_name}")

                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()

                for col in columns:
                    _, col_name, col_type, _, _, _ = col
                    schema_info.append(f"  - {col_name} ({col_type})")

                schema_info.append("")  # Blank line between tables

            return "\n".join(schema_info)

        except Exception as e:
            return f"Error retrieving schema: {str(e)}"

        finally:
            conn.close()

    def normalize_sql(self, sql):
        """Remove extraneous formatting, code block markers, and standardize SQL."""
        sql = sql.strip().replace("```sql", "").replace("```", "").strip()  # Remove markdown SQL blocks
        return sqlparse.format(sql, reindent=True, keyword_case='upper').strip()  # Standardize format