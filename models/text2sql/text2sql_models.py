import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import anthropic
from config.config import Config
if Config.debug:
    import pdb

def create_text2sql_func_openai(model_name="gpt-3.5-turbo", temperature=0, categories=None):
    """
    Creates a function that generates SQL queries from natural language questions 
    using OpenAI's API with a specified LLM model.
    Parameters:
        model_name (str): The OpenAI language model to use (e.g., "gpt-3.5-turbo").
        temperature (float, optional): Between 0 and 1.

    Returns:
        function: A function `question_to_sql(question, schema_info)` that generates
                  SQL queries given a natural language question and database schema.

    """
    _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
    OpenAI.api_key = os.environ['OPENAI_API_KEY']  # Set API key from environment
    client = OpenAI()

    def question_to_sql(question, schema_info):
        """
        Generates an SQL query from a natural language question using OpenAI's API.

        Parameters:
            question (str): The natural language question to convert into SQL.
            schema_info (str): The database schema information, including table 
                               and column definitions.

        Returns:
            str: The generated SQL query.
        """
        # prompt = f"""
        # You are an AI that converts natural language questions into SQL queries.
        # Use the provided database schema to generate the correct SQL query.
        # Make sure the SQL you generate is specifically for an **SQLite3** database.
        # Database Schema:
        # {schema_info}

        # Question: "{question}"
        # Expected Output: Provide a valid **SQLite3-compatible** SQL query based on the question and schema.
        # SQL Query:
        # """

        prompt = f"""
        You are an AI that converts natural language questions into SQL queries 
        specifically for an **SQLite3** database. 

        ### **Important SQLite Constraints:**
        - Use **only INNER JOIN, LEFT JOIN, or CROSS JOIN** (no FULL OUTER JOIN).
        - **No native JSON functions** (assume basic text handling).
        - Data types are flexible; prefer **TEXT, INTEGER, REAL, and BLOB**.
        - **BOOLEAN is represented as INTEGER** (0 = False, 1 = True).
        - Use **LOWER()** for case-insensitive string matching.
        - Primary keys auto-increment without `AUTOINCREMENT` unless explicitly required.
        - Always assume **foreign key constraints are disabled unless explicitly turned on**.

        ### **Database Schema:**
        {schema_info}

        ### **User Question:**
        "{question}"

        ### **Expected Output:**
        Provide a valid **SQLite3-compatible** SQL query based on the question and schema.
        SQL Query:
        """

        # categories_str = ", ".join(f"'{c}'" for c in categories)

        # prompt = f"""
        # You are an AI that converts natural language questions into SQL queries 
        # specifically for an **SQLite3** database. 

        # ### **Important SQLite Constraints:**
        # - Use **only INNER JOIN, LEFT JOIN, or CROSS JOIN** (no FULL OUTER JOIN).
        # - **No native JSON functions** (assume basic text handling).
        # - Data types are flexible; prefer **TEXT, INTEGER, REAL, and BLOB**.
        # - **BOOLEAN is represented as INTEGER** (0 = False, 1 = True).
        # - Use **LOWER()** for case-insensitive string matching.
        # - Primary keys auto-increment without `AUTOINCREMENT` unless explicitly required.
        # - Always assume **foreign key constraints are disabled unless explicitly turned on**.

        # ### **Domain Knowledge:**
        # - The `category` column in the `object_detections` table has a **fixed set of known values**.
        # - These values are: {categories_str}.
        # - All category comparisons in the query should use **exact string equality** (e.g. `category = 'Vehicle'`) and not partial matches or `LIKE`.
        # - You can assume that values in the `category` column are clean, normalized, and always match one of the known options.
        # - If an aspect of the question can map onto a category, use the exact string equality comparison.

        # ### **Database Schema:**
        # {schema_info}

        # ### **User Question:**
        # "{question}"

        # ### **Expected Output:**
        # Provide a valid **SQLite3-compatible** SQL query based on the question and schema.
        # SQL Query:
        # """

        messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )

        return response.choices[0].message.content.strip()

    return question_to_sql


def create_text2sql_func_deepseek(model_name = None, temperature=0):
    """
    Creates a function that generates SQL queries from natural language questions 
    using DeepSeek's API with a specified LLM model.
    Parameters:
        temperature (float, optional): Between 0 and 1.

    Returns:
        function: A function `question_to_sql(question, schema_info)` that generates
                  SQL queries given a natural language question and database schema.

    """
    _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
    # OpenAI.api_key = os.environ['OPENAI_API_KEY']  # Set API key from environment
    client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")

    def question_to_sql(question, schema_info):
        """
        Generates an SQL query from a natural language question using OpenAI's API.

        Parameters:
            question (str): The natural language question to convert into SQL.
            schema_info (str): The database schema information, including table 
                               and column definitions.

        Returns:
            str: The generated SQL query.
        """
        # prompt = f"""
        # You are an AI that converts natural language questions into SQL queries.
        # Use the provided database schema to generate the correct SQL query.
        # Make sure the SQL you generate is specifically for an **SQLite3** database.
        # Database Schema:
        # {schema_info}

        # Question: "{question}"
        # Expected Output: Provide a valid **SQLite3-compatible** SQL query based on the question and schema.
        # SQL Query:
        # """

        # prompt = f"""
        # You are an AI that converts natural language questions into SQL queries 
        # specifically for an **SQLite3** database. 

        # ### **Important SQLite Constraints:**
        # - Use **only INNER JOIN, LEFT JOIN, or CROSS JOIN** (no FULL OUTER JOIN).
        # - **No native JSON functions** (assume basic text handling).
        # - Data types are flexible; prefer **TEXT, INTEGER, REAL, and BLOB**.
        # - **BOOLEAN is represented as INTEGER** (0 = False, 1 = True).
        # - Use **LOWER()** for case-insensitive string matching.
        # - Primary keys auto-increment without `AUTOINCREMENT` unless explicitly required.
        # - Always assume **foreign key constraints are disabled unless explicitly turned on**.

        # ### **Database Schema:**
        # {schema_info}

        # ### **User Question:**
        # "{question}"

        # ### **Expected Output:**
        # Provide a valid **SQLite3-compatible** SQL query based on the question and schema.
        # SQL Query:
        # """
        prompt = prompt = f"""
        ### System Instruction ###
        Convert the following question into a **valid SQLite3 SQL query**.
        - **Only return the SQL query, without any explanation**.
        - **Do not include extra comments or explanations**.
        - **End the SQL query with a semicolon (`;`)**.

        **Schema**:
        {schema_info}

        **User Question**:
        {question}

        **SQL Query**:
        """

        messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=temperature,
        )
        generated_sql = response.choices[0].message.content.strip()
        # **Stop output if it contains an explanation**
        if "### Explanation" in generated_sql:
            generated_sql = generated_sql.split("### Explanation")[0].strip()

        return generated_sql

    return question_to_sql


def create_text2sql_func_anthropic(model_name="claude-3-5-haiku-latest", temperature=0):
    """
    Creates a function that generates SQL queries from natural language questions 
    using Anthropic's API with a specified LLM model.
    Parameters:
        model_name (str): The OpenAI language model to use (e.g., "gpt-3.5-turbo").
        temperature (float, optional): Between 0 and 1.

    Returns:
        function: A function `question_to_sql(question, schema_info)` that generates
                  SQL queries given a natural language question and database schema.

    """
    _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
    client = anthropic.Anthropic(
        api_key=os.environ['ANTHROPIC_API_KEY'],
    )

    def question_to_sql(question, schema_info):
        """
        Generates an SQL query from a natural language question using OpenAI's API.

        Parameters:
            question (str): The natural language question to convert into SQL.
            schema_info (str): The database schema information, including table 
                               and column definitions.

        Returns:
            str: The generated SQL query.
        """
        # prompt = f"""
        # You are an AI that converts natural language questions into SQL queries.
        # Use the provided database schema to generate the correct SQL query.
        # Make sure the SQL you generate is specifically for an **SQLite3** database.
        # Database Schema:
        # {schema_info}

        # Question: "{question}"
        # Expected Output: Provide a valid **SQLite3-compatible** SQL query based on the question and schema.
        # SQL Query:
        # """

        prompt = f"""
        You are an AI that converts natural language questions into SQL queries 
        specifically for an **SQLite3** database. 

        ### **Important SQLite Constraints:**
        - Use **only INNER JOIN, LEFT JOIN, or CROSS JOIN** (no FULL OUTER JOIN).
        - **No native JSON functions** (assume basic text handling).
        - Data types are flexible; prefer **TEXT, INTEGER, REAL, and BLOB**.
        - **BOOLEAN is represented as INTEGER** (0 = False, 1 = True).
        - Use **LOWER()** for case-insensitive string matching.
        - Primary keys auto-increment without `AUTOINCREMENT` unless explicitly required.
        - Always assume **foreign key constraints are disabled unless explicitly turned on**.

        ### **Database Schema:**
        {schema_info}

        ### **User Question:**
        "{question}"

        ### **Expected Output:**
        Provide a valid **SQLite3-compatible** SQL query based on the question and schema.
        **Do not include explanations or extra text.**
        Return only the SQL query
        SQL Query:
        """

        messages = [{"role": "user", "content": prompt}]

        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=messages,
        )

        return response.content[0].text.strip()

    return question_to_sql