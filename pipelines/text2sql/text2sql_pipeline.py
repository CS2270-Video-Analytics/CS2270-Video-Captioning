import sqlite3
import ast
import re
import sqlparse
import sqlglot
from sqlglot import exp
import logging
import asyncio
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

    def clear_pipeline(self):
        #clear the cache that remains for previous runs of text2sql
        self.table_attributes = []
        self.all_objects = []
    
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
                is_sufficient, existing_tables_attributes_dict, new_tables_attributes_dict, sql_query = await self.check_schema_sufficiency_llm_judge(query=question, table_schemas = table_schemas)
            else:
                #NOTE: required_tables isn't currently being used but can be integrated if we want to create new tables
                is_sufficient, existing_tables_attributes_dict, new_tables_attributes_dict, sql_query = await self.check_schema_sufficiency_manual(query=question, table_schemas = table_schemas)
            
            return is_sufficient, sql_query, existing_tables_attributes_dict, new_tables_attributes_dict
        except Exception as e:
            raise RuntimeError(f"Error in text2sql run inference: {e}")
    
    async def check_schema_sufficiency_manual(self, query: str, table_schemas: str):
        prompt = Config.get_text2sql_prompt(table_schemas, query)
        sql_query = await self.text2sql_model.run_inference(prompt)
        
        sufficient, existing_tables_attributes_dict, new_tables_attributes_dict = self.parse_schema_manually(sql_query, table_schemas)
        
        return sufficient, existing_tables_attributes_dict, new_tables_attributes_dict, sql_query

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
        existing_tables_attributes_dict = {}
        new_tables_attributes_dict = {}
        for table in sql_schema_dict:
            if table not in table_schema_dict:
                new_tables_attributes_dict[table] = sql_schema_dict[table]
                sufficient = False
            else:
                existing_tables_attributes_dict[table] = sql_schema_dict[table].difference(table_schema_dict[table])
                if len(existing_tables_attributes_dict[table]) > 0:
                    sufficient = False
        return sufficient, existing_tables_attributes_dict, new_tables_attributes_dict


    def extract_sql_schema_dict(self, sql_query: str):
        try:
            parsed = sqlglot.parse_one(sql_query)
            alias_to_table = {}
            table_columns = {}
            
            # Extract all tables from the query
            all_tables = [t.name for t in parsed.find_all(exp.Table)]
            default_table = all_tables[0] if all_tables else "unknown_table"

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
                    alias = node.table
                    column = node.name
                    
                    # We need to find the actual table for columns with no table reference
                    if alias is None or alias == "":
                        # Use the default table from our earlier extraction
                        alias = default_table
                    
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
                # Handle empty string as alias (replace with default table)
                if true_table == "":
                    true_table = default_table
                resolved.setdefault(true_table, set()).update(cols)

            result = {table: set(cols) for table, cols in resolved.items()}
            
            # If we have an empty string key, map it to the default table
            if "" in result:
                if default_table not in result:
                    result[default_table] = result[""]
                else:
                    result[default_table].update(result[""])
                del result[""]
                
            return result
            
        except Exception as e:
            logger.error(f"Error parsing SQL query '{sql_query}': {str(e)}")
            
            # Fallback to simple regex parsing if sqlglot fails
            tables = re.findall(r'FROM\s+([a-zA-Z0-9_]+)', sql_query, re.IGNORECASE)
            columns = re.findall(r'SELECT\s+(?:DISTINCT\s+)?(.+?)\s+FROM', sql_query, re.IGNORECASE)
            
            if tables and columns:
                table_name = tables[0]
                column_list = [col.strip() for col in columns[0].split(',')]
                where_columns = re.findall(r'WHERE\s+([a-zA-Z0-9_]+)', sql_query, re.IGNORECASE)
                if where_columns:
                    column_list.extend(where_columns)
                return {table_name: set(column_list)}
            
            raise RuntimeError(f"Failed to parse SQL query: {sql_query}") from e

    async def check_schema_sufficiency_llm_judge(self, query: str, table_schemas: str):
        prompt = Config.schema_sufficiency_prompt.format(query=query, table_schemas=table_schemas)
        
        sql_query = None
        for _ in range(Config.max_schema_sufficiency_retries):
            sufficiency_response = await self.text2sql_model.run_inference(prompt)
            if "error" in sufficiency_response.lower():
                raise RuntimeError(f"Error in text2sql run inference: {sufficiency_response}")
            try:
                sufficient, existing_tables_attributes_dict, new_tables_attributes_dict = self.parse_llm_judge_response(sufficiency_response)
            except Exception as e:
                raise RuntimeError(f"Error in check_schema_sufficiency_llm_judge: {e}") from e
            
            if sufficient and existing_tables_attributes_dict:
                prompt = Config.get_text2sql_prompt(table_schemas, query)
                try:
                    sql_query = await self.text2sql_model.run_inference(prompt)
                except Exception as e:
                    raise RuntimeError(f"Error in text2sql run inference: {e}") from e
                return sufficient, existing_tables_attributes_dict, new_tables_attributes_dict, sql_query
        
        return sufficient, existing_tables_attributes_dict, new_tables_attributes_dict, sql_query
    
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
            if existing_line is None:
                existing_tables_attributes_dict = None
            else:
                start_index = existing_line.index(":") + 1
                end_index = len(existing_line)
                existing_tables_attributes_dict = existing_line[start_index:end_index].strip()
                existing_tables_attributes_dict = re.sub(r"(\w+)", r"'\1'", existing_tables_attributes_dict)
                existing_tables_attributes_dict = ast.literal_eval(existing_tables_attributes_dict)
            
            create_line = next((line for line in lines if line.lower().startswith("new")), None)
            if create_line is None:
                new_tables_attributes_dict = None
            else:
                start_index = create_line.index(":") + 1
                end_index = len(create_line)
                new_tables_attributes_dict = create_line[start_index:end_index].strip()
                new_tables_attributes_dict = re.sub(r"(\w+)", r"'\1'", new_tables_attributes_dict)
                new_tables_attributes_dict = ast.literal_eval(new_tables_attributes_dict)
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Error in parse_llm_judge_response: {e}") from e
        
        return sufficiency, existing_tables_attributes_dict, new_tables_attributes_dict
