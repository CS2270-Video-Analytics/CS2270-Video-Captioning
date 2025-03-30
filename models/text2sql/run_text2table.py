import sqlite3
import sqlparse
import re
import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

def create_table(db_file, output_db_file):
    schema = get_schema(db_file, "raw_videos", num_rows=5)
    print("Schema:\n", schema)
    create_sqlite3_db(schema, output_db_file)

def clean_schema(schema: str) -> str: 
    lines = schema.strip().splitlines() 
    cleaned_lines = [line for line in lines if not re.match(r"^\s*```.*$", line)] 
    return "\n".join(cleaned_lines)

def get_schema(db_path, table_name, num_rows=40):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # SQL query to get combined description
    query = """
        SELECT GROUP_CONCAT(description, ' ') AS combined_description
        FROM (
            SELECT description FROM raw_videos LIMIT ?
        )
    """

    cursor.execute(query, (num_rows,))
    result = cursor.fetchone()

    combined_description = result[0] if result[0] else ''
    # print("Combined Description:\n", combined_description)

    # Close the connection
    conn.close()

    return get_schema_openai(combined_description, "gpt-4o-mini", "general street scene")

def get_schema_openai(combined_description, model_name, topic, temperature=0.2):
    _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
    OpenAI.api_key = os.environ['OPENAI_API_KEY']  # Set API key from environment
    client = OpenAI()

    prompt = f"""
    Sample text :
    < tr class =" mergedrow " > < th scope =" row " class =" infobox -
    label " > < div style =" text - indent : -0.9 em ; margin - left
    :1.2 em ; font - weight : normal ;" > < a href ="/ wiki /
    Monarchy_of_Canada " title =" Monarchy of Canada " >
    Monarch </ a > </ div > </ th > < td class =" infobox - data " > <
    a href ="/ wiki / Charles_III " title =" Charles III " >
    Charles III </ a > </ td > </ tr >
    < tr class =" mergedrow " > < th scope =" row " class =" infobox -
    label " > < div style =" text - indent : -0.9 em ; margin - left
    :1.2 em ; font - weight : normal ;" > < span class =" nowrap
    " > < a href ="/ wiki / Governor_General_of_Canada "
    title =" Governor General of Canada " > Governor
    General </ a > </ span > </ div > </ th > < td class =" infobox -
    data " > < a href ="/ wiki / Mary_Simon " title =" Mary
    Simon " > Mary Simon </ a > </ td > </ tr >
    <b > Provinces and Territories </ b class =' navlinking
    countries '>
    <ul >
    <li > Saskatchewan </ li >
    <li > Manitoba </ li >
    <li > Ontario </ li >
    <li > Quebec </ li >
    <li > New Brunswick </ li >
    <li > Prince Edward Island </ li >
    <li > Nova Scotia </ li >
    <li > Newfoundland and Labrador </ li >
    <li > Yukon </ li >
    <li > Nunavut </ li >
    <li > Northwest Territories </ li >
    </ ul >
    Question : List all relevant attributes about 'Canada '
    that are exactly mentioned in this sample text if
    any .
    Answer :
    - Monarch : Charles III
    - Governor General : Mary Simon
    - Provinces and Territories : Saskatchewan , Manitoba ,
    Ontario , Quebec , New Brunswick , Prince Edward
    Island , Nova Scotia , Newfoundland and Labrador ,
    Yukon , Nunavut , Northwest Territories
    ----
    Sample text :
    Patient birth date : 1990 -01 -01
    Prescribed medication : aspirin , ibuprofen ,
    acetaminophen
    Prescribed dosage : 1 tablet , 2 tablets , 3 tablets
    Doctor 's name : Dr . Burns
    Date of discharge : 2020 -01 -01
    Hospital address : 123 Main Street , New York , NY 10001
    Question : List all relevant attributes about '
    medications ' that are exactly mentioned in this
    sample text if any .
    Answer :
    - Prescribed medication : aspirin , ibuprofen ,
    acetaminophen
    - Prescribed dosage : 1 tablet , 2 tablets , 3 tablets
    ----
    Sample text :
    {combined_description}
    Question : List all relevant attributes about '{topic
    } ' that are exactly mentioned in this sample
    text if any .
    Answer :
    """

    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )

    print("Response:\n", response.choices[0].message.content.strip())
    print()
    print("---------------------------------------------------------")
    print()

    prompt2 = f"""
    Based on the following attributes, design a Database schema for a Sqlite3 Database with the
    attribute names as column headers.

    Each table you create MUST have a column named "frame_id" which is should be the PRIMARY KEY and of TYPE TEXT.

    ONLY RETURN THE SQLITE3 TABLE CREATION STATEMENT FOR ALL THE TABLES
    DO NOT RETURN ANY OTHER TEXT

    Attributes:
    {response.choices[0].message.content.strip()}
    Database schema:
    """


    messages = [{"role": "user", "content": prompt2}]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()

def create_sqlite3_db(schema, db_file):
    # Connect to the SQLite database
    schema = clean_schema(schema)
    print()
    print("---------------------------------------------------------")
    print()
    print("Schema FOR SQL:\n", schema)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Execute the schema SQL to create the table
    try:
        cursor.executescript(schema)
        conn.commit()
        print(f"Database '{db_file}' created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating database: {e}")
    finally:
        conn.close()

def populate_table(db_file, output_db_file):
    original_schema = get_existing_schema(db_file)
    print("Original Schema:\n", original_schema)
    output_db_schema = get_existing_schema(output_db_file)
    print("Schema:\n", output_db_schema)
    insert_into_table(db_file, output_db_file, output_db_schema)


def get_existing_schema(db_file):
    """
    Extracts the schema information from all tables in an SQLite database.

    Parameters:
        db_file (str): Path to the SQLite database file.

    Returns:
        str: The schema of all tables, including column details.
    """
    import sqlite3

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        # Get all table names from sqlite_master
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        all_tables = [row[0] for row in cursor.fetchall()]

        if not all_tables:
            return "No tables found in the database."

        schema_info = []

        for table_name in all_tables:
            schema_info.append(f"Table: {table_name}")

            # Get column details
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            for col in columns:
                _, col_name, col_type, _, _, _ = col
                schema_info.append(f"  - {col_name} ({col_type})")

            schema_info.append("")  # Add space between tables

        return "\n".join(schema_info)

    except Exception as e:
        return f"Error retrieving schema: {str(e)}"

    finally:
        conn.close()

def insert_into_table(db_file, output_db_file, output_db_schema):
    # Function to parse CREATE TABLE schema into a dict
    def parse_schema(schema_str):
        table_defs = {}
        table_blocks = re.findall(r"CREATE TABLE (\w+)\s*\((.*?)\);", schema_str, re.DOTALL)
        for table_name, columns_str in table_blocks:
            columns = []
            for line in columns_str.strip().splitlines():
                line = line.strip().strip(",")  # clean up
                if line:
                    col_parts = line.split()
                    col_name = col_parts[0]
                    columns.append(col_name)
            table_defs[table_name] = columns
        return table_defs

    # Parse schema to extract table and column structure
    parsed_schema = parse_schema(output_db_schema)

    # Connect and create the destination DB
    dst_conn = sqlite3.connect(output_db_file)
    dst_cursor = dst_conn.cursor()
    # dst_cursor.executescript(output_db_schema)

    # Connect to the source DB
    src_conn = sqlite3.connect(db_file)
    src_cursor = src_conn.cursor()
    src_cursor.execute("SELECT frame_id, description FROM raw_videos")
    rows = src_cursor.fetchall()

    # Prompt template
    prompt_template = """
    You are an expert at converting natural language descriptions of traffic scenes into structured data.

    Given the following description:

    "{description}"

    Extract the structured information and return it as JSON using this format:

    {{
    {json_schema}
    }}

    Only include fields that can be reasonably inferred. Leave out objects that are not mentioned. Use null for missing but relevant fields.
    """

    # Function to build JSON format string from schema
    def build_json_template(schema_dict, frame_id_placeholder="{frame_id}"):
        lines = []
        for table, cols in schema_dict.items():
            lines.append(f'  "{table}": [')
            lines.append("    {")
            for col in cols:
                if col == "frame_id":
                    value = f'"{frame_id_placeholder}"'
                else:
                    value = '...'
                lines.append(f'      "{col}": {value},')
            lines[-1] = lines[-1].rstrip(",")  # remove trailing comma
            lines.append("    }")
            lines.append("  ],")
        lines[-1] = lines[-1].rstrip(",")  # remove trailing comma
        return "\n".join(lines)

    print("---------------------------------------------------------")
    print("Parsed Schema:\n", parsed_schema)
    json_schema_template = build_json_template(parsed_schema)

    # LLM call function
    def call_llm(prompt):
        _ = load_dotenv(find_dotenv())  # Load environment variables from .env file
        OpenAI.api_key = os.environ['OPENAI_API_KEY']  # Set API key from environment
        client = OpenAI()
        response = client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message["content"]

    # Process rows
    for frame_id, description in rows:
        prompt = prompt_template.format(description=description, json_schema=json_schema_template).replace("{frame_id}", str(frame_id))

        try:
            structured_json = call_llm(prompt)
            parsed = json.loads(structured_json)

            for table, columns in parsed_schema.items():
                records = parsed.get(table, [])
                if not isinstance(records, list):
                    continue

                for record in records:
                    values = [record.get(col) for col in columns]
                    placeholders = ", ".join(["?"] * len(columns))
                    column_names = ", ".join(columns)
                    insert_sql = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"
                    dst_cursor.execute(insert_sql, values)

            dst_conn.commit()

        except Exception as e:
            print(f"Error processing frame_id {frame_id}: {e}")

    # Clean up
    src_conn.close()
    dst_conn.close()