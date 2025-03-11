import json
import os
from tqdm import tqdm
from text2sql_pipeline import get_schema, execute_sql, normalize_sql
from text2sql_openai import create_text2sql_func_openai
from text2sql_hf import create_text2sql_func_hf
 
def load_test_data(file_path, limit=5):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit]  # Return first limit entries

def evaluate_text2sql(test_data, db_base_path, text2sql_func, to_print=False):
    correct = 0
    total = len(test_data)
    
    for entry in tqdm(test_data, desc="Evaluating", unit="query"):
        question = entry["question"]
        ground_truth_sql = entry["query"]
        db_id = entry["db_id"]
        db_file = os.path.join(db_base_path, db_id, f"{db_id}.sqlite")
        
        # Generate SQL using the pipeline
        schema = get_schema(db_file)
        generated_sql = normalize_sql(text2sql_func(question, schema))
        
        # Execute both SQL queries
        ground_truth_result = execute_sql(db_file, ground_truth_sql)
        generated_result = execute_sql(db_file, generated_sql)
        
        # Compare results
        if ground_truth_result == generated_result:
            correct += 1
        else:
            if to_print:
                print(f"Mismatch in db_id: {db_id}")
                print(f"Mismatch for question: {question}")
                print(f"Ground Truth SQL: {ground_truth_sql}")
                print(f"Generated SQL: {generated_sql}")
                print(f"Ground Truth Result: {ground_truth_result}")
                print(f"Generated Result: {generated_result}\n")
    
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    TEST_JSON_PATH = os.path.join(BASE_DIR, "datasets", "Spider_test", "test.json")
    DB_PATH = os.path.join(BASE_DIR, "datasets", "Spider_test", "test_database")
    
    # text2sql_func = create_text2sql_func_openai("gpt-3.5-turbo")
    # text2sql_func = create_text2sql_func_openai("gpt-4o-mini")
    text2sql_func = create_text2sql_func_hf("gaussalgo/T5-LM-Large-text2sql-spider")
    test_data = load_test_data(TEST_JSON_PATH, limit=25)
    
    evaluate_text2sql(test_data, DB_PATH, text2sql_func, to_print=False)
