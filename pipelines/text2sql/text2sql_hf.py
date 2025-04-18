import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

def create_text2sql_func_hf(model_name):
    """
    Creates a function that converts natural language questions into SQL queries
    using the Hugging Face model_name running locally.

    Parameters:
        model_name (str): The Hugging Face model to use.

    Returns:
        function: A function `text2sql(question, schema_info)` that generates
                  SQLite-compatible SQL queries.
    """
    # Load tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "OpenELM" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    if "T5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    def text2sql(question, schema_info):
        """
        Generates an SQLite-compatible SQL query from a natural language question.

        Parameters:
            question (str): The natural language question.
            schema_info (str): The database schema information.

        Returns:
            str: The generated SQL query.
        """
        # prompt = f"translate English to SQL: {question} | Schema: {schema_info}"
        prompt = f"""
        translate English to SQL:
        - Use **SQLite3** syntax.
        - Use **only INNER JOIN, LEFT JOIN, or CROSS JOIN** (no FULL OUTER JOIN).
        - **No JSON functions** (assume basic text handling).
        - Data types are flexible; prefer **TEXT, INTEGER, REAL, and BLOB**.
        - **BOOLEAN should be INTEGER** (0 = False, 1 = True).
        - Use **LOWER()** for case-insensitive string matching.
        - Primary keys auto-increment without `AUTOINCREMENT` unless explicitly required.
        - Assume **foreign key constraints are disabled** unless explicitly turned on.

        **Schema**: {schema_info}

        **User Question**: {question}

        **SQL Query**:
        """
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate SQL query
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                num_beams=5,
                pad_token_id=tokenizer.eos_token_id
                )
                # temperature=0.3, pad_token_id=tokenizer.eos_token_id)

        # Decode and return the generated SQL
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # if "SQL Query:" in generated_sql:
        #     generated_sql = generated_sql.split("SQL Query:")[-1].strip()
        #         # Stop at "-- END QUERY" if it exists
        # if "-- END QUERY" in generated_sql:
        #     generated_sql = generated_sql.split("-- END QUERY")[0].strip()

        return generated_sql

    return text2sql
