import torch

class Config:
    llama_model_name = 'llama3.2-vision:11b'
    clip_model_name = "ViT-L/14"
    model_precision = torch.float16
    system_eval = False
    obj_focus = False
    previous_frames = True
    max_tokens = 300
    
    question_prompt_format = "Question: {question} Answer:"
    sliding_window_caption_prompt_format = "Here are descriptions of previous frames from a video.\nDescription of previous frames:{prev_frames_context}\nDescribe every individual object in the current image with unique object ids, as a continuation to previous image descriptions. Give descriptions of each object's attributes, the action each object performs and its relation to other objects:"
    object_extraction_pormpt_format = "Current image description: {curr_img_caption}\nObjects (with unique IDs) seen so far: {obj_list}\nGiven the current image description and list of objects, identify any new objects (with their object IDs) not seen so far." 
    sliding_window_size = 3

    #models to be used for captioning
    caption_model = 'LLamaVision'

    #only for debugging
    debug = True

    # batch size for inserting into the DB
    batch_size = 40
    frames_per_video = 40

    #schema definition for databases
    caption_schema = dict(video_id = "INTEGER UNIQUE NOT NULL", frame_id = "INTEGER UNIQUE NOT NULL", description="TEXT NOT NULL", objects="TEXT")
    table_name = "videos"
    base_db_path = '/users/ssunda11/git/CS2270-Video-Captioning/database_integration'
    db_name = "video_frames.db"