import sqlite3
import os
from config import Config
from typing import List, Dict, Set, Optional
import pdb
import logging
# Set up logging
logger = logging.getLogger(__name__)

class SQLLiteDBInterface:
    def __init__(self, db_name: str, table_name_schema_dict: Optional[Dict] = None):
        # Connect to SQLite database (or create it if it doesn't exist)
        self.connection = sqlite3.connect(os.path.join(Config.sql_db_path, Config.sql_db_name if db_name is None else db_name))
        self.cursor = self.connection.cursor()

        if not table_name_schema_dict:
            self.table_name_schema_dict = {Config.caption_table_name: [Config.caption_table_schema, Config.caption_table_pk]}
        else:
            self.table_name_schema_dict = table_name_schema_dict
        
        #create the table during the init
        self.create_table()
        #otherwiese populate the table schema dictionary from before
        if os.path.exists(os.path.join(Config.sql_db_path, Config.sql_db_name if db_name is None else db_name)):
            self.table_name_schema_dict = self.extract_schema_dict()

        self.insertion_query = "INSERT OR IGNORE INTO {table_name} {table_schema} VALUES ({table_schema_value})"
        self.add_col_query = "ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type} DEFAULT NULL;"

    def create_table(self):
        for (table_name, vals) in self.table_name_schema_dict.items():
            [schema, primary_keys] = vals 
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {','.join(key + ' ' + val for (key, val) in schema.items())},
                    PRIMARY KEY ({','.join(primary_keys)})
                ) 
            ''')
            self.connection.commit()

    def add_new_table(self, table_name:str, table_schema:str, table_prim_key:str):
        self.table_name_schema_dict[table_name] = [table_schema, table_prim_key]
        self.create_table()

    def extract_schema_dict(self):
        # Get all user-defined table names (exclude internal SQLite tables)
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in self.cursor.fetchall()]

        table_name_schema_dict = {}

        for table in tables:
            self.cursor.execute(f"PRAGMA table_info({table});")
            rows = self.cursor.fetchall()

            # Format: (cid, name, type, notnull, dflt_value, pk)
            column_dict = {}
            primary_keys = []

            for cid, name, col_type, notnull, default, pk in rows:
                column_dict[name] = col_type
                if pk > 0:
                    primary_keys.append(name)

            table_name_schema_dict[table] = [column_dict, primary_keys]

        return table_name_schema_dict

    def execute_query(self, query: str, args: Optional[ Optional[tuple]] = None):
        """
        Executes an SQL query on a given SQLite database.

        Parameters:
            query (str): The SQL query to execute.
            args (tuple): Additional arguments to SQL query
            
        Returns:
            list: Query result as a list of tuples, or an error message (str) if execution fails.
        """
        try:
            if not args:
                self.cursor.execute(query)
            else:
                self.cursor.execute(query, args)

            result = self.cursor.fetchall()
            logger.debug(f"SQL execution result: {result}")
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return f"Error: {str(e)}"

        return result
    
    def execute_script(self, queries: str):
        try:
            self.cursor.executescript(queries)
            self.connection.commit()
            self.table_name_schema_dict = self.extract_schema_dict() #update the schema dictionary if we create new tables
            return self.cursor.fetchall()  # return all rows relevant to query
        except Exception as e:
            print(f"Error executing multiple queries: {e}")
            return None
    #NOTE: this is not used anymore but may be needed for future use
    def create_column(self, table_name: str, col_name: str, col_type: str):
        try:
            #first alter table to create new column
            add_col_query = self.add_col_query.format(table_name=table_name, col_name=col_name, col_type=col_type)
            self.cursor.execute(add_col_query)
            self.connection.commit()
        except Exception as e:
            if "duplicate column name" in e:
                pass
            else:
                raise RuntimeError(f"Error in insert_columns: {e}")

    def insert_column_data(self, table_name:str, col_name:str, col_type:str, data:list):
        try:
            # 2. Create temporary update table
            self.cursor.execute("DROP TABLE IF EXISTS temp;")
            self.cursor.execute(f"CREATE TEMP TABLE temp (id INTEGER, id INTEGER, new_val {col_type});")

            # 3. Insert values into temp
            self.cursor.executemany("INSERT INTO temp (id, id, new_val) VALUES (?, ?, ?);", data)

            # 4. Update using a join
            self.cursor.execute(f"""
                UPDATE {table_name}
                SET {col_name} = (
                    SELECT new_val FROM temp WHERE temp.id = {table_name}.id
                )
                WHERE id IN (SELECT id FROM temp);
            """)
            self.cursor.execute("DROP TABLE IF EXISTS temp;")
            # 5. Finalize
            self.conn.commit()
        except Exception as e:
            print(f"Error inserting columns {col_name} into table {table_name}")
            return None

    def insert_many_rows_list(self, table_name:str, rows_data: list):
        table_schema = self.table_name_schema_dict[table_name]
        schema_prompt = tuple(table_schema[0].keys())
        schema_value_prompt = ','.join(['?' for _ in range(len(table_schema[0].keys()))])
        
        query = self.insertion_query.format(table_name = table_name, table_schema = schema_prompt, table_schema_value = schema_value_prompt)
        
        try:
            # Insert multiple rows at once using executemany()
            self.cursor.executemany(query, rows_data)
            self.connection.commit()
        except Exception as e:
            print(f"Error inserting {rows_data} into {table_name}: {e}")

    def extract_all_rows(self, table_name:str):
        return self.execute_query(query=f"SELECT * FROM {table_name};")
    
    def get_total_num_rows(self, table_name:str):
        #get total number of rows in table
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        num_rows = int(self.cursor.fetchone()[0])
        return num_rows

    def extract_concatenated_captions(self, table_name:str, attribute:str, num_rows:int = 40):
        # SQL query to get combined description
        query = f"""
            SELECT GROUP_CONCAT({attribute}, ' ') AS combined_{attribute}
            FROM (
                SELECT {attribute} FROM {table_name} LIMIT ?
            )
        """

        self.cursor.execute(query, (num_rows,))
        result = self.cursor.fetchone()
        combined_description = result[0] if result[0] else ''

        return combined_description

    def close_conn(self):
        self.cursor.close()
    
    def get_table_schema(self, table_name: str) -> str:
        """
        Extracts the schema information for a specific table in the SQLite database.

        Parameters:
            table_name (str): Name of the table to retrieve the schema for.

        Returns:
            str: The schema of the specified table, including column details.
        """
        try:
            # Check if the table exists
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            if not self.cursor.fetchone():
                return f"Table '{table_name}' does not exist in the database."

            # Get column details
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = self.cursor.fetchall()
            schema_info = [f"Table: {table_name}"]
            for col in columns:
                col_id, col_name, col_type, _, _, _ = col
                schema_info.append(f"  - {col_name} ({col_type})")

            return "\n".join(schema_info)
        except Exception as e:
            return f"Error retrieving schema for table '{table_name}': {str(e)}"

    def get_all_schemas_except_raw_videos(self) -> str:
        """
        Extracts the schema information for all tables in the SQLite database except 'raw_videos'.

        Returns:
            str: The schemas of all tables except 'raw_videos', including column details.
        """
        try:
            # Get all user-defined table names except 'raw_videos'
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'raw_videos';")
            tables = [row[0] for row in self.cursor.fetchall()]

            if not tables:
                return "No tables exist in the database except 'raw_videos'."

            schema_info = []
            for table in tables:
                schema_info.append(self.get_table_schema(table))
            return "\n".join(schema_info)

        except Exception as e:
            return f"Error retrieving schemas: {str(e)}"
