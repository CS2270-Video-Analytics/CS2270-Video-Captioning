# CS2270-Video-Captioning

## Project Overview
This project is a modular video captioning and querying system. It processes videos to generate captions, converts captions to structured SQL tables, and enables natural language or SQL-based queries over the extracted data. The system integrates state-of-the-art vision-language models, text-to-SQL pipelines, and a modern web frontend for video upload and querying.

## Directory Structure
- `pipeline.py`: **Backend entry point**. Orchestrates the full video-to-table and query pipeline.
- `models/`: Submodules for vision-language models, language models, and text-to-SQL models. Includes configuration files and model wrappers.
- `datasets/`: References to all datasets (raw data) used for training, evaluation, and testing.
- `data_processing/`: Scripts and configs for preprocessing and sampling videos.
- `outputs/`: Contains the generated PostgreSQL tables and processed outputs.
- `frontend/`: Flask app and HTML/JS interface for uploading videos and querying data.
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
### 1. Run the Backend Pipeline
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

### 2. Run the Frontend
```bash
cd frontend
python app.py
```
Then open `frontend/index.html` in your browser. Use the interface to upload videos and submit queries.

## Profiling
Profile backend latency using Pyinstrument:
```bash
pyinstrument -o test_async_latency_text2table.html pipeline.py
```

---
For questions or contributions, please open an issue or contact the maintainers.