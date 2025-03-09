# CS2270-Video-Captioning

## Overall Structure
* `models' : contains all submodules or model classes for captioning, text to SQL and text to table + config files
* `datasets': contains reference to all datasets (raw data) stored
* `data_processing': code for preprocessing and sampling videos + config files
* `outputs': contains the Postgre SQL table generated from our system
* `frontend': contains code to setup and run front-end for video uploading
* `main.py': the initial entrypoint for code backend containing logic to select models and run pipeline
* `baselines': contains all submodules and code for baseline models e.g. VIVA and ZELDA