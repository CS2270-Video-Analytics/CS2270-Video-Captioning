import sqlite3
import sqlparse
import logging
from config.config import Config
if Config.debug:
    import pdb
from models.text2sql import Text2SQLModelFactory
# Set up logging
logger = logging.getLogger(__name__)

class Text2SQLPipeline():
    """Pipeline for converting natural language questions to SQL queries and executing them"""

    def __init__(self):
        """
        Initialize the Text2SQL pipeline with the model specified in the config
        """
        try:
            # Parse model type and name from config
            [text2sql_model, text2sql_model_name] = Config.text2sql_model_name.split(';')
            
            # Create the model using the factory
            self.model = Text2SQLModelFactory.create_model(text2sql_model, text2sql_model_name)
            logger.info(f"Initialized Text2SQL pipeline with model: {text2sql_model}:{text2sql_model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Text2SQL pipeline: {str(e)}")
            raise
    
    def check_schema_sufficiency(self, question: str, table_schema: str):
        last_result = None
        prompt = Config.schema_sufficiency_prompt.format(question=question, schema_info=table_schema)
        for _ in range(Config.max_schema_sufficiency_retries):
            sufficiency_response = self.model(prompt)
            sufficiency, required_attributes = self.parse_schema_sufficiency_response(sufficiency_response)
            last_result = (sufficiency, required_attributes)
            if sufficiency == "Yes":
                return sufficiency, required_attributes
            elif sufficiency == "No":
                return sufficiency, required_attributes
        # Return the result from the last run
        return last_result

    def parse_schema_sufficiency_response(self, response: str) -> tuple[str, list[str]]:
        """
        Parses the LLM schema sufficiency response.

        Args:
            response (str): The raw response string from the LLM.

        Returns:
            Tuple[str, List[str]]: ("Yes" or "No", list of required attribute strings)
        """
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        
        # First line: Sufficient: Yes/No
        sufficiency_line = next((line for line in lines if line.lower().startswith("sufficient:")), None)
        if not sufficiency_line:
            raise ValueError("Missing 'Sufficient:' line in response.")
        
        sufficiency = sufficiency_line.split(":")[1].strip()

        # Find start of 'Required Attributes' section
        try:
            start_index = lines.index("Required Attributes:") + 1
        except ValueError:
            raise ValueError("Missing 'Required Attributes:' section in response.")
        
        required_attributes = lines[start_index:]

        return sufficiency, required_attributes

    def run_pipeline(self, question: str, table_schema: str):
        """
        Converts a natural language question into an SQL query.

        Parameters:
            question (str): The natural language question to convert to SQL.
            table_schema (str): The database schema information.

        Returns:
            str: The generated SQL query.
        """
        try:
            prompt = Config.get_text2sql_prompt(table_schema, question)
            sql_query = self.normalize_sql(self.model(prompt))
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

if __name__ == "__main__":
    # Simple test setup
    test_schema = """
    CREATE TABLE cars (
        id INTEGER PRIMARY KEY,
        make TEXT,
        model TEXT,
        year INTEGER,
        color TEXT,
        price REAL
    );
    """
    
    # Initialize pipeline
    pipeline = Text2SQLPipeline()
    
    # Test questions
    test_questions = [
        "Show me all red cars",
        "What is the average price of cars?",
        "Find cars made after 2020",
        "List all Toyota models sorted by price"
    ]
    
    print("\n=== Testing Text2SQL Pipeline ===")
    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            sql_query = pipeline.run_pipeline(question, test_schema)
            print(f"Generated SQL: {sql_query}")
            
            # Optional: Test execution with a sample database
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            
            # Create test table and insert sample data
            cursor.execute(test_schema)
            sample_data = [
                (1, 'Toyota', 'Corolla', 2020, 'Blue', 25000),
                (2, 'Honda', 'Civic', 2021, 'Red', 27000),
                (3, 'Toyota', 'Camry', 2019, 'Red', 30000)
            ]
            cursor.executemany(
                'INSERT INTO cars VALUES (?, ?, ?, ?, ?, ?)',
                sample_data
            )
            
            # Execute generated query
            print("Executing query...")
            cursor.execute(sql_query)
            results = cursor.fetchall()
            print(f"Results: {results}")
            
            conn.close()
            
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)