import os
import logging
import openai
from config.config import Config

# Set up logging
logger = logging.getLogger(__name__)

class Text2SQLModelFactory:
    """Factory class for creating text-to-SQL model functions"""
    
    @staticmethod
    def create_model(model_type, model_name=None):
        """
        Create a text-to-SQL model function based on the specified model type
        
        Args:
            model_type (str): The type of model to create (e.g., 'OpenAI', 'DeepSeek', etc.)
            model_name (str, optional): The specific model name to use
            
        Returns:
            function: A function that converts natural language to SQL
            
        Raises:
            ValueError: If the model type is not supported
        """
        model_creators = {
            'OpenAI': Text2SQLModelFactory._create_openai_model,
            'DeepSeek': Text2SQLModelFactory._create_deepseek_model,
            'Anthropic': Text2SQLModelFactory._create_anthropic_model,
            'HuggingFace': Text2SQLModelFactory._create_huggingface_model
        }
        
        if model_type not in model_creators:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(model_creators.keys())}")
        
        return model_creators[model_type](model_name)
    
    @staticmethod
    def _create_openai_model(model_name=None):
        """Create a function that uses OpenAI models for text-to-SQL conversion"""
        
        # Use model name from config if not provided
        if model_name is None:
            model_name = Config.text2sql_model_name.split(';')[1]
        
        # Set up OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        openai.api_key = api_key
        
        def text2sql_func(question, schema_info):
            """
            Convert natural language question to SQL using OpenAI model
            
            Args:
                question (str): The natural language question to convert
                schema_info (str): Information about the database schema
                
            Returns:
                str: The generated SQL query
                
            Raises:
                Exception: If there's an error with the API call
            """
            try:
                # Generate the prompt using the config
                prompt = Config.get_text2sql_prompt(schema_info, question)
                
                # Get parameters from config
                params = Config.text2sql_params
                
                # Call the OpenAI API with all parameters from config
                request_params = dict(
                    model=model_name,
                    messages = [{"role": "user", "content": prompt}],
                    temperature=params['temperature'],
                    max_tokens=params['max_tokens'],
                    top_p=params['top_p'],
                    frequency_penalty=params['frequency_penalty'],
                    presence_penalty=params['presence_penalty'],
                    stop=params['stop_tokens']  # optional
                )

                # Filter out any that are None
                request_params = {k: v for k, v in request_params.items() if v is not None}

                # Send the request
                response = openai.chat.completions.create(**request_params)
                
                # Extract the SQL query from the response
                sql_query = response.choices[0].message.content.strip()
                return sql_query
                
            except Exception as e:
                logger.error(f"Error in OpenAI text2sql: {str(e)}")
                raise
        
        return text2sql_func
    
    @staticmethod
    def _create_deepseek_model(model_name=None):
        """Create a function that uses DeepSeek models for text-to-SQL conversion"""
        # This is a placeholder - implement with actual DeepSeek API when available
        def text2sql_func(question, schema_info):
            raise NotImplementedError("DeepSeek model implementation not available yet")
        return text2sql_func
    
    @staticmethod
    def _create_anthropic_model(model_name=None):
        """Create a function that uses Anthropic models for text-to-SQL conversion"""
        # This is a placeholder - implement with actual Anthropic API when available
        def text2sql_func(question, schema_info):
            raise NotImplementedError("Anthropic model implementation not available yet")
        return text2sql_func
    
    @staticmethod
    def _create_huggingface_model(model_name=None):
        """Create a function that uses HuggingFace models for text-to-SQL conversion"""
        # This is a placeholder - implement with actual HuggingFace API when available
        def text2sql_func(question, schema_info):
            raise NotImplementedError("HuggingFace model implementation not available yet")
        return text2sql_func

# For backward compatibility
def create_text2sql_func_openai(model_name=None):
    """Create a function that uses OpenAI models for text-to-SQL conversion"""
    return Text2SQLModelFactory._create_openai_model(model_name)

def create_text2sql_func_deepseek(model_name=None):
    """Create a function that uses DeepSeek models for text-to-SQL conversion"""
    return Text2SQLModelFactory._create_deepseek_model(model_name)

def create_text2sql_func_anthropic(model_name=None):
    """Create a function that uses Anthropic models for text-to-SQL conversion"""
    return Text2SQLModelFactory._create_anthropic_model(model_name)

def create_text2sql_func_hf(model_name=None):
    """Create a function that uses HuggingFace models for text-to-SQL conversion"""
    return Text2SQLModelFactory._create_huggingface_model(model_name) 