import sqlite3
import sqlparse
import logging
from config.config import Config
if Config.debug:
    import pdb
from models.text2sql import Text2SQLModelFactory
from models.language_models.OpenAIText import OpenAIText
from models.language_models.OllamaText import OllamaText
from models.language_models.Anthropic import Anthropic
from models.language_models.DeepSeek import DeepSeek
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

    def run_pipeline(self, question: str, table_schemas: str, llm_judge:bool=True):
        """
        Converts a natural language question into an SQL query using the existing database schema.

        Parameters:
            question (str): The natural language question to convert to SQL.
            db_file (str): Path to the SQLite database file.

        Returns:
            str: The generated SQL query.
        """
        try:
            sufficiency, required_attributes = self.check_schema_sufficiency(query=question, table_schemas = table_schemas, llm_judge=llm_judge)
            if sufficiency:
                prompt = Config.get_text2sql_prompt(table_schemas, question)
                sql_query, info = self.text2sql_model.run_inference(prompt)
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

    def check_schema_sufficiency(self, query: str, table_schemas: str, llm_judge:bool=True):
        if llm_judge:
            prompt = Config.schema_sufficiency_prompt.format(query=query, table_schemas=table_schemas)
            
            for _ in range(Config.max_schema_sufficiency_retries):
                sufficiency_response, info = self.text2sql_model.run_inference(prompt)
                if info['error']:
                    raise Exception(info['error'])

                sufficient, required_attributes = self.parse_schema_sufficiency_response_llm_judge(sufficiency_response)
                if sufficient and required_attributes:
                    return sufficient, None

        return sufficient, required_attributes
    
    def parse_schema_sufficiency_response_llm_judge(self, response: str) -> tuple[str, list[str]]:
        """
        Parses the LLM schema sufficiency response.
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
            start_index = lines.index("Required Attributes:") + 1
            required_attributes = lines[start_index:]
        except ValueError:
            required_attributes = None

        return sufficiency, required_attributes

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