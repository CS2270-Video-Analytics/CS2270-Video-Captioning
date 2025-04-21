import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import pandas as pd
from pipelines.text2sql.text2sql_pipeline import Text2SQLPipeline
from database_integration import SQLLiteDBInterface
from config.config import Config
import asyncio
from tqdm import tqdm
import csv

# Load dataset
df = pd.read_csv("datasets/eval_datasets/video_annotations_accuracy.csv")

# Initialize pipeline
pipeline = Text2SQLPipeline()
results = []

# Correctness checking
def compute_correctness(comparator, ground_truth, system_output):
    if comparator == "equals":
        return int(str(system_output) == str(ground_truth))
    elif comparator == "range":
        try:
            return int(abs(float(system_output) - float(ground_truth)) <= 1)
        except:
            return 0
    elif comparator == "IoU":
        try:
            gt_set = set(s.strip() for s in str(ground_truth).split(","))
            sys_set = set(s.strip() for s in str(system_output).split(","))
            intersection = gt_set & sys_set
            union = gt_set | sys_set
            return len(intersection) / len(union) if union else 0
        except:
            return 0
    return 0

original_video_filename = Config.video_filename  # Cache the clean base
sql_db_path = './database_integration'

async def main():
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        nl_query = row['nl_query']
        dataset = row['dataset']
        video_id = row['video_id']
        category = row['category']
        ground_truth = row['ground_truth']
        comparator = row['comparator']

        if video_id != Config.video_filename:
            Config.video_filename = video_id
            Config.sql_db_path = sql_db_path
            Config.sql_db_name = Config.video_filename + ".db"
            Config.db_path = os.path.join(Config.sql_db_path, Config.sql_db_name)
            print(Config.db_path)
            text2sql_pipeline = Text2SQLPipeline()
            sql_dbs = SQLLiteDBInterface(Config.sql_db_name)

        try:
            start_time = time.time()
            table_schemas = sql_dbs.get_all_schemas_except_raw_videos()

            sql_query = await text2sql_pipeline.run_pipeline(
                question=nl_query, 
                table_schemas=table_schemas, 
                llm_judge=False
            )
            result = sql_dbs.execute_query(query=sql_query)
            processing_time = time.time() - start_time

            if not result:
                system_output = ""
            elif len(result) == 1 and len(result[0]) == 1:
                system_output = result[0][0]
            else:
                system_output = ",".join(str(row[0]) for row in result)

            correctness = compute_correctness(comparator, ground_truth, system_output)

        except Exception as e:
            processing_time = time.time() - start_time
            sql_query = "ERROR"
            system_output = "ERROR"
            correctness = 0

        results.append({
            "dataset": dataset,
            "video_id": video_id,
            "category": category,
            "nl_query": nl_query,
            "ground_truth": ground_truth,
            "comparator": comparator,
            "SQL_query": str(sql_query),
            "system_output": str(system_output),
            "processing_time": processing_time,
            "correctness": correctness
        })

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    #results_df.to_csv("datasets/eval_results/video_annotation_evaluation_results.csv", index=False, quoting=csv.QUOTE_ALL)

# Run async main
asyncio.run(main())
