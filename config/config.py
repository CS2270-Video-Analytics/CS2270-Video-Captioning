import torch

class Config:
    vision_model_name = 'llama3.2-vision:11b'
    text_model_name = 'llama3.2:latest'
    clip_model_name = "ViT-L/14"
    model_precision = torch.float8
    system_eval = False
    obj_focus = False
    previous_frames = True
    max_tokens = 300
    keep_alive = 30000
    batch_size = 5
    num_threads = 8
    
    question_prompt_format = "Question: {question} Answer:"
    sliding_window_caption_prompt_format = "Here are descriptions of previous frames from a video.\nDescription of previous frames:{prev_frames_context}\nDescribe every individual object in the current image with unique object ids, as a continuation to previous image descriptions. Give descriptions of each object's attributes, the action each object performs and its relation to other objects:"
    object_extraction_pormpt_format = "Current image description: {curr_img_caption}\nObjects (with unique IDs) seen so far: {obj_list}\nGiven the current image description and list of objects, identify any new objects (with their object IDs) not seen so far." 
    sliding_window_size = 3

    text2table_attribute_extraction_prompt = "All image captions:{incontext_captions}\nGiven captions of all images, generate a list of important attributes that can describe the objects in the image: [color, brand, size, action, speed, location, position, state]\n\nAll image captions:{frame_captions}\nGiven captions of all images, generate a list of important attributes that can describe objects in the image:"
    text2table_incontext_prompt = "\n-".join(["A red Nissan SUV is in the center of the road, putting on the breaks, whilst a young female pedestrian wearing yellow shirt crosses", "A red Nissan SUV driving to the left lane and a grey Chevrolet and blue motorbike on the right lane. The motorbike is speeding without breaks"])
    text2table_frame_prompt = "Given a detailed description of an image and the set of objects below, output a structured table with the columns {formatted_schema}.  Separate table rows with <r> and table columns with <c>\n\nobject: one of the objects in set of objects\nattributes: detailed descriptive qualities about the object and its properties; where attribute is not present, leave empty\nimage_location: spatial location of object in the image relative to other objects\naction: the detailed process or movement the object is performing; where action is not relevant for an object, leave empty\n\nDetailed image description:{image_caption}\n\nSet of objects:{object_set}"
    #model to be used for text2table
    text2table_model = 'Ollama' #options: [Ollama, GPT3, Seq2Seq]

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