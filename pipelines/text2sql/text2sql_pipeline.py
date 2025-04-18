import sqlite3
import ast
import re
import sqlparse
import sqlglot
from sqlglot import exp
import logging
from config.config import Config
if Config.debug:
    import pdb
from models.text2sql import Text2SQLModelFactory
from models.language_models.OpenAIText import OpenAIText
from models.language_models.OllamaText import OllamaText
from models.language_models.Anthropic import Anthropic
from models.language_models.DeepSeek import DeepSeek
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
# Set up logging
logger = logging.getLogger(__name__)

class Text2SQLPipeline():
    """Pipeline for converting natural language questions to SQL queries and executing them"""

    def __init__(self):
        """
        Initialize the Text2SQL pipeline with the model specified in the config
        """
        try:
            #initialize the model that needs to be used for captioning
            model_options = {'Ollama': OllamaText, 'OpenAI': OpenAIText, 'Anthropic': Anthropic, 'DeepSeek':DeepSeek}
            [text2sql_model, text2sql_model_name] = Config.caption_model_name.split(';')
            assert text2sql_model in model_options, f'ERROR: model {text2sql_model} does not exist or is not supported yet'
            
            self.text2sql_model = model_options[text2sql_model](
                                    model_params = Config.text2sql_params, 
                                    model_name=text2sql_model_name, 
                                    model_precision=Config.model_precision, 
                                    system_eval=Config.system_eval)

            logger.info(f"Initialized Text2SQL pipeline with: {self.text2sql_model.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Text2SQL pipeline: {str(e)}")
            raise

    async def run_pipeline(self, question: str, table_schemas: str, llm_judge: bool=True):
        """
        Converts a natural language question into an SQL query using the existing database schema.

        Parameters:
            question (str): The natural language question to convert to SQL.
            db_file (str): Path to the SQLite database file.

        Returns:
            str: The generated SQL query.
        """
        try:
            if llm_judge:
                sufficiency, required_attributes = self.check_schema_sufficiency_llm_judge(query=question, table_schemas = table_schemas)
            else:
                #NOTE: required_tables isn't currently being used but can be integrated if we want to create new tables
                sufficiency, required_attributes, required_tables, sql_query, info = self.check_schema_sufficiency_manual(query=question, table_schemas = table_schemas)
            if sufficiency and not llm_judge:
                prompt = Config.get_text2sql_prompt(table_schemas, question)
                sql_query, info = await self.text2sql_model.run_inference(prompt)
                if info['error']:
                    raise Exception(info['error'])
                logger.debug(f"Generated SQL query: {sql_query}")
                return sql_query
            else:
                raise NotImplementedError("SQL Table Update: Text2Column Module to be implemented")
        except Exception as e:
            logger.error(f"Error in run_pipeline: {str(e)}")
            raise
    
    def clear_pipeline(self):
        #clear the cache that remains for previous runs of text2sql
        self.table_attributes = []
        self.all_objects = []
    
    def check_schema_sufficiency_manual(self, query: str, table_schemas: str):
        prompt = Config.get_text2sql_prompt(table_schemas, query)
        try:
            sql_query, info = self.text2sql_model.run_inference(prompt)
            if info['error']:
                raise Exception(info['error'])
            logger.debug(f"Generated SQL query: {sql_query}")
        except Exception as e:
            logger.error(f"Error in run_pipeline: {str(e)}")
            raise
        
        sufficient, required_attributes, required_tables = self.parse_schema_manually(sql_query, table_schemas)
        if sufficient and required_attributes:
            return sufficient, required_attributes, required_tables, sql_query, info
        else:
            return sufficient, required_attributes, required_tables, sql_query, info

    def extract_sql_schema_dict(self, sql_query: str):
        parsed = sqlglot.parse_one(sql_query)
        alias_to_table = {}
        table_columns = {}

        def collect_aliases(node):
            """Collect alias → table mappings from FROM and JOIN clauses."""
            if isinstance(node, exp.From) or isinstance(node, exp.Join):
                for source in node.find_all(exp.Table):
                    alias = source.alias_or_name
                    table = source.name
                    alias_to_table[alias] = table

            for child in node.args.values():
                if isinstance(child, list):
                    for c in child:
                        if isinstance(c, exp.Expression):
                            collect_aliases(c)
                elif isinstance(child, exp.Expression):
                    collect_aliases(child)

        def collect_columns(node):
            """Collect columns grouped under their alias or table reference."""
            if isinstance(node, exp.Column):
                alias = node.table or "<unknown>"
                column = node.name
                table_columns.setdefault(alias, set()).add(column)

            for child in node.args.values():
                if isinstance(child, list):
                    for c in child:
                        if isinstance(c, exp.Expression):
                            collect_columns(c)
                elif isinstance(child, exp.Expression):
                    collect_columns(child)

        # Step 1: Get alias → table
        collect_aliases(parsed)
        # Step 2: Get columns used
        collect_columns(parsed)
        # Step 3: Normalize alias-based keys to actual table names
        resolved = {}
        for alias_or_table, cols in table_columns.items():
            true_table = alias_to_table.get(alias_or_table, alias_or_table)
            resolved.setdefault(true_table, set()).update(cols)

        return {table: set(cols) for table, cols in resolved.items()}

    def parse_schema_manually(self, sql_query: str, table_schemas: str):
        """
        Parses the schema manually to determine a sufficiency response.
        Args:
            sql_query (str): The sql query string to extract sufficiency information
            table_schemas (str): All table schemas/columns as a string to be parsed
        Returns:
            Tuple[str, List[str]]: ("Yes" or "No", list of required attribute strings)
        """
        #parse raw table schemas into dictionary format
        table_schema_dict = {}
        current_table = None

        for line in table_schemas.strip().splitlines():
            line = line.strip()
            # Match Table: <table_name>
            if line.startswith("Table:"):
                current_table = line.split("Table:")[1].strip()
                table_schema_dict[current_table] = set([])
            # Match column definitions
            elif current_table and line:
                # Extract column name, ignoring datatype
                col_match = re.search(r'-\s*(\w+)\s*\(', line)
                if col_match:
                    column_name = col_match.group(1)
                    table_schema_dict[current_table].add(column_name)
        
        #parse SQL statement into dictionary 
        sql_schema_dict = self.extract_sql_schema_dict(sql_query=sql_query)

        #compare the raw table schema with the parsed sql schema and find missing attributes and/or tables
        sufficient = True
        required_attributes = {}
        required_tables = {}
        for table in sql_schema_dict:
            if table not in table_schema_dict:
                required_tables[table] = sql_schema_dict[table]
                sufficient = False
            else:
                required_attributes[table] = sql_schema_dict[table].difference(table_schema_dict[table])
                if len(required_attributes[table]) > 0:
                    sufficient = False
        return sufficient, required_attributes, required_tables

    def check_schema_sufficiency_llm_judge(self, query: str, table_schemas: str):
        prompt = Config.schema_sufficiency_prompt.format(query=query, table_schemas=table_schemas)
        
        for _ in range(Config.max_schema_sufficiency_retries):
            sufficiency_response, info = self.text2sql_model.run_inference(prompt)
            if info['error']:
                raise Exception(info['error'])

            sufficient, required_attributes = self.parse_llm_judge_response(sufficiency_response)
            if sufficient and required_attributes:
                return sufficient, None
        
        return sufficient, required_attributes
    
    def parse_llm_judge_response(self, response: str) -> tuple[str, list[str]]:
        """
        Parses the schema using LLM to determine a sufficiency response.
        Args:
            response (str): The raw response string from the LLM.
        Returns:
            Tuple[str, List[str]]: ("Yes" or "No", list of required attribute strings)
        """
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

        #check for sufficient in output
        sufficiency_line = next((line for line in lines if line.lower().startswith("sufficient:")), None)
        if not sufficiency_line:
            sufficiency = None
        elif sufficiency_line.split(":")[1].strip() == "Yes":
            sufficiency = True
        elif sufficiency_line.split(":")[1].strip() == "No":
            sufficiency = False
        else:
            sufficiency = None

        #check required attributes section
        try:
            existing_line = next((line for line in lines if line.lower().startswith("existing")), None)
            start_index = existing_line.index(":") + 1
            end_index = len(existing_line)

            required_attributes = existing_line[start_index:end_index].strip()
            required_attributes = re.sub(r"(\w+)", r"'\1'", required_attributes)
            required_attributes = ast.literal_eval(required_attributes)
            
            create_line = next((line for line in lines if line.lower().startswith("new")), None)
            start_index = create_line.index(":")+1
            end_index = len(create_line)
            
            new_attributes = create_line[start_index:end_index].strip()
            new_attributes = re.sub(r"(\w+)", r"'\1'", new_attributes)
            new_attributes = ast.literal_eval(new_attributes)
            if new_attributes == '\'None\'':
                new_attributes = None

        except ValueError:
            required_attributes = None

        return sufficiency, required_attributes, new_attributes

if __name__ == "__main__":
    pipeline = Text2SQLPipeline()
    test_questions = [
        # "Find all video_id and frame_id where someone is standing",
        # "How many males are there wearing a black shirt in the video?",
        # "Are there any cabinet in the video?",
        # "What color is the cabinet in the video?"
        "Find all cars appeared in the video with clear enough license plates"
        "Find the frame traffic light turned green"
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
        