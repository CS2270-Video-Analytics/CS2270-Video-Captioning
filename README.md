# CS2270-Video-Captioning

## System Overview

This system enables users to query the content of videos using natural language by automatically extracting structured information from video frames and storing it in a relational database. The system is adaptive: if a user's query cannot be answered with the current database schema, it will automatically update the schema and extract the missing attributes in the form of new tables or/and new columns.

### 1. Before Query: Video Processing & Database Construction

- **Frame Sampling:**  
  Input videos are sampled into individual frames at a configurable rate.
- **Frame Captioning:**  
  Each frame is processed by a vision-language model, which generates detailed captions describing all visible objects, their categories, attributes (e.g., brand, color), actions, and locations.
- **Table Generation:**  
  The system parses these captions and organizes the extracted information into SQL tables. Each object and its attributes are stored in a structured, queryable format. The database schema is generated dynamically based on the detected object types and their properties.

---

### 2. User Query: Natural Language to SQL

- **Query Submission:**  
  Users submit natural language queries (e.g., "Find all frames where a white BMW is turning right") through the web frontend.
- **Text-to-SQL Translation:**  
  The system uses a language model to translate the user's query into an SQL statement, leveraging the current database schema.
- **Result Retrieval:**  
  The SQL query is executed, and the relevant frames or information are returned to the user.

---

### 3. If the Query Can't Be Answered: Schema Adaptation & Re-kicking

- **Schema Sufficiency Check:**  
  If the system determines that the current database schema is missing key attributes or tables needed to answer the query, it invokes an LLM-based "Judge" to assess what's missing.
- **Schema Extension:**  
  - If new object categories or attributes are required, the system will:
    - **Generate new tables** if entirely new table(s) are needed.
    - **Add new columns** to existing tables if only additional attribute(s) are required.
- **Re-kicking the Pipeline:**  
  The system then "re-kicks" the pipeline: it extracts the missing information from the video, updates the database, and automatically re-attempts the user's query.

## Directory Structure
- `pipeline.py`: **Backend entry point**. Orchestrates the full video-to-table and query pipeline.
- `models/`: Submodules for vision-language models, language models, and text-to-SQL models. Includes configuration files and model wrappers.
- `datasets/`: References to all datasets (raw data) used for training, evaluation, and testing.
- `data_processing/`: Scripts and configs for preprocessing and sampling videos.
- `outputs/`: Contains the generated PostgreSQL tables and processed outputs.
- `frontend/`: Flask app and HTML/JS interface for uploading videos and querying data. WIP.
- `baselines/`: Baseline models (e.g., VIVA, ZELDA) for comparison.
- `database_integration/`: Interfaces for SQLite and vector database integration.
- `pipelines/`: Modular pipeline components for frame extraction, captioning, text-to-SQL, text-to-table, and text-to-column.
- `requirements.txt`: Python dependencies for the project.

## Setup
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Request GPU from OSCAR (if needed)
```bash
srun --partition=gpu --gres=gpu:2 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
```

## Edit the Config file
Before running the pipeline, **edit `config/config.py`** to set the appropriate options for your environment and task. Key settings include:

- **Input video path and filename:**
  ```python
  video_path = 'datasets/bdd/'
  video_filename = '00afa5b2-c14a542f.mov'
  ```
- **Model selection:**
  ```python
  caption_model_name = 'OpenAI;gpt-4o-mini'
  text2table_model_name = 'OpenAI;gpt-4o-mini'
  text2sql_model_name = 'OpenAI;gpt-4o-mini'
  ```
- **Database paths and names**
- **Pipeline toggles and parameters** (e.g., batch size, precision, enabling/disabling modules)

## Usage
### Run the Backend Pipeline
Edit `pipeline.py` as needed for your use case (see the `__main__` section for examples). Example to process a query for a missing table:
```python
# In pipeline.py
query_pipeline = VideoQueryPipeline()
# Example: Query for a missing table (advanced use case)
import time
question = "What is the first frame in which a damaged SUV stops at a red light?"
start_time = time.time()
result = asyncio.run(query_pipeline.process_query(language_query=question, llm_judge=Config.llm_judge))
end_time = time.time()
print("SYSTEM RESPONSE: ", result)
print(f"Time taken: {end_time - start_time}")
```


## Profiling
Profile backend latency using Pyinstrument:
```bash
pyinstrument -o test_async_latency_text2table.html pipeline.py
```

---
For questions or contributions, please open an issue or contact the maintainers.