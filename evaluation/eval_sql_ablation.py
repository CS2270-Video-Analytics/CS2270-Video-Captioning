import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import pandas as pd
from database_integration import SQLLiteDBInterface
from pipelines.text2sql.text2sql_pipeline import Text2SQLPipeline
from config.config import Config
from tqdm import tqdm
import csv

async def main():
    # Load test dataset
    df = pd.read_csv("datasets/eval_datasets/SQL_ablation.csv")

    # Ensure correct config path
    sql_db_path = './database_integration/db_files/'

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        dataset = row['dataset']
        video_id = row['video_id']
        nl_query = row['nl_query']
        gt_SQL = row['gt_SQL']
        expected_value = row['ground_truth']

        if video_id != Config.video_filename:
            Config.video_filename = video_id
            Config.sql_db_path = sql_db_path
            Config.sql_db_name = Config.video_filename + ".db"
            Config.db_path = os.path.join(Config.sql_db_path, Config.sql_db_name)
            print(f"Using DB: {Config.db_path}")
            sql_dbs = SQLLiteDBInterface(Config.sql_db_name)
            text2sql_pipeline = Text2SQLPipeline()

        try:
            gt_result = sql_dbs.execute_query(query=gt_SQL)

            start_time = time.time()
            table_schemas = sql_dbs.get_all_schemas_except_raw_videos()

            _, system_query, _, _ = await text2sql_pipeline.run_pipeline(
                question=nl_query, 
                table_schemas=table_schemas, 
                llm_judge=False
            )
            system_result = sql_dbs.execute_query(query=system_query)
            processing_time = time.time() - start_time
            gt_result = gt_result[0][0]
            if str(gt_result) != str(expected_value):
                raise ValueError(f"Ground truth SQL result does not match expected value: {gt_result} != {expected_value} for {gt_SQL} for {nl_query}")
            
            system_output = system_result[0][0]
            correctness = int(gt_result == system_output)

            # For logging purposes, join flattened output into strings
            ground_truth_str = str(gt_result)
            system_output_str = str(system_output)

        except Exception as e:
            print(f"Error processing row: {e}")
            query_latency = time.time() - start_time
            ground_truth_str = "ERROR"
            system_output_str = "ERROR"
            correctness = 0

        results.append({
            "dataset": dataset,
            "video_id": video_id,
            "nl_query": nl_query,
            "gt_SQL": gt_SQL,
            "ground_truth": ground_truth_str,
            "query_latency": processing_time,
            "system_query": system_query,
            "system_output": system_output_str,
            "correctness": correctness
        })

    # Save results
    results_df = pd.DataFrame(results)
    # results_df.to_csv("datasets/eval_results/sql_ablation_results.csv", index=False, quoting=csv.QUOTE_ALL)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
