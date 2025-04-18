import time
import pandas as pd
import ast
from pipelines.text2sql.text2sql_pipeline import Text2SQLPipeline
from config.config import Config

# Initialize pipeline
pipeline = Text2SQLPipeline()

# Initialize results
results = []

# Function to evaluate precision and recall
class Metrics:
    @staticmethod
    def calculate_precision_recall(expected_labels, query_results):
        true_positives = 0
        for frame_id in query_results:
            # Check if the frame_id is within any of the specified ranges
            if any(start <= frame_id <= end for start, end in frame_ranges):
                true_positives += 1
        precision = true_positives / len(query_results) if query_results else 0
        recall = true_positives / len(expected_labels) if expected_labels else 0
        return precision, recall


# Initialize pipeline
pipeline = Text2SQLPipeline()
verb_label_range = pd.read_csv('./verb_label_range.csv')
# Initialize results
results = []

# Process each cleaned data entry
for index, entry in verb_label_range.iterrows():
    video_id = entry['video_id']
    verb = entry['verb']
    frame_ranges = ast.literal_eval(entry['ranges'])
    
    # Generate SQL query using the pipeline
    question = f"Find all frame_id where someone is performing an action of '{verb}'"
    sql_query = pipeline.run_pipeline(question, Config.db_path)
    
    # Execute the query to get results
    query_results = pipeline.execute_sql(Config.db_path, sql_query) # video_id, frame_id pairs
    query_frame_ids = [result[1] for result in query_results] 
    
    # Calculate precision and recall
    precision, recall = Metrics.calculate_precision_recall(frame_ranges, query_frame_ids)
    
    # Append results
    results.append({
        'video_id': video_id,
        'query': verb,
        'precision': precision,
        'recall': recall
    })

for result in results:
    print(f"Video ID: {result['video_id']}, Query: {result['query']}, Precision: {result['precision']}, Recall: {result['recall']}")
