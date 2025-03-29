import sqlite3
import sqlparse

def text2sql_pipeline(question, db_file, text2sql_func):
    """
    Converts a natural language question into an SQL query and executes it on a given SQLite database.

    Parameters:
        question (str): The natural language question to convert to SQL.
        db_file (str): Path to the SQLite database file.
        text2sql_func (function): A function that generates SQL queries given a question and schema.

    Returns:
        tuple: The generated SQL query and its execution result.
    """
    db_schema = get_schema(db_file)
    sql_query = normalize_sql(text2sql_func(question, db_schema))
    sql_output = execute_sql(db_file, sql_query)
    return sql_query, sql_output

def execute_sql(db_file, sql_query):
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

def get_schema(db_file):
    """
    Extracts the schema information from an SQLite database.

    Parameters:
        db_file (str): Path to the SQLite database file.

    Returns:
        str: The database schema, including table names and column details.
    """
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

def normalize_sql(sql):
    """Remove extraneous formatting, code block markers, and standardize SQL."""
    sql = sql.strip().replace("```sql", "").replace("```", "").strip()  # Remove markdown SQL blocks
    return sqlparse.format(sql, reindent=True, keyword_case='upper').strip()  # Standardize format