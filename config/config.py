import torch

class Config:
    vision_model_name = 'llama3.2-vision:11b'
    text_model_name = 'llama3.2:latest'
    clip_model_name = "ViT-L/14"
    model_precision = torch.float8
    system_eval = False
    obj_focus = True
    previous_frames = True
    max_tokens = 300
    keep_alive = 30000
    batch_size = 5
    num_threads = 8
    
    question_prompt_format = "Question: {question} Answer:"
    # sliding_window_caption_prompt_format = "Here are descriptions of previous frames from a video.\nDescription of previous frames:{prev_frames_context}\nDescribe every individual object in the current image with unique object ids, as a continuation to previous image descriptions. Give descriptions of each object's attributes, the action each object performs and its relation to other objects:"
    
    sliding_window_caption_prompt_format = """
    Task: given the current driving scene frame recorded from a dash cam angle, for each unique key object in the frame, output the object's id, category, attributes, action and location exactly using general template's format below. So say if we have 4 key objects then the output should be 4 chunks of 5-line output like below separted by new line. Don't include anything else such as the image shows xxx or image summary.

    General template:
    object id: the object's id starting from 1; restart from 1 for any new frame
    category: object's category
    attributes: object's identifying attributes of the category it belongs. If vehicle it can be vehicle type, brand, model, color, license plate, etc. For example, if pedestrian it can be gender, height, outfit, race, etc. If traffic light it can be current color of the light. Only include if they are identifiable and visible enough.
    action: object's state or action inferred. If it occurs also in {prev_frames_context}, infer the action based on the difference of both frames.
    location: object's relative position to other key objects in the frame

    Example (for some non-existent frame):
    object id: 1
    category: vehicle
    attributes: brand BMW, color white, license plate LW-527, type SUV
    action: parking
    location: right of the road

    object id: 2
    category: traffic light
    attributes: red
    action: red
    location: overhead

    object id: 3
    category: pedestrian
    attributes: tall, yellow hat
    action: walking
    location: right of the street

    Task (for the current frame):
    object id:"""

    object_extraction_prompt_format = """
    Task: given current frame description and seen object categories so far, output a list of unique unseen new object categories from the current frame's description that are not seen before. Don't return me the code as I want the actual the output.

    General template:
    current frame description:
        object: object's id
        category: object's category (try to reuse same name from {obj_list} if they are the same)
        attributes: object's identifying attributes of the category it belongs; include attribute only if clearly visible
        action: object's action
        location: object's relative location to other key objects in the current frame
    seen object category: a list of unique seen object categories before
    unseen objects: a list of unseen object categories in the current frame

    Example 1 (non-existent):
    current frame description:
        object: 1
        category: vehicle
        attributes: brand BMW, color white, license plate LW-527, type SUV
        action: parking
        location: right of the road

        object: 2
        category: vehicle
        attributes: brand Benz, color green, type truck
        action: driving
        location: in the middle of the road

    seen objects: []
    unseen objects: [vehicle]

    Example 2 (non-existent):
    current frame description:
        object: 1
        category: traffic light
        attributes: red
        action: red
        location: overhead

        object: 2
        category: pedestrian
        attributes: male, tall, asian
        action: walking
        location: right of the street
    seen objects: [vehicle]
    unseen objects: [traffic light, pedestrian]

    Example 3 (non-existent):
    current frame description:
        object: 1
        category: traffic light
        attributes: green
        action: green
        location: overhead
    seen objects: [vehicle, traffic light, pedestrian]
    unseen objects: []

    Task (for the current frame):
    current frame description: {curr_img_caption}
    seen objects: [{obj_list}]
    unseen objects:"""

    sliding_window_size = 1

    text2table_attribute_extraction_prompt = "All image captions:{incontext_captions}\nGiven captions of all images, generate a list of important attributes that can describe the objects in the image: [color, brand, size, action, speed, location, position, state]\n\nAll image captions:{frame_captions}\nGiven captions of all images, generate a list of important attributes that can describe objects in the image:"
    text2table_incontext_prompt = "\n-".join(["A red Nissan SUV is in the center of the road, putting on the breaks, whilst a young female pedestrian wearing yellow shirt crosses", "A red Nissan SUV driving to the left lane and a grey Chevrolet and blue motorbike on the right lane. The motorbike is speeding without breaks"])
    text2table_frame_prompt = "Given a detailed description of an image and the set of objects below, output a structured table with the columns {formatted_schema}.  Separate table rows with <r> and table columns with <c>\n\nobject: one of the objects in set of objects\nattributes: detailed descriptive qualities about the object and its properties; where attribute is not present, leave empty\nimage_location: spatial location of object in the image relative to other objects\naction: the detailed process or movement the object is performing; where action is not relevant for an object, leave empty\n\nDetailed image description:{image_caption}\n\nSet of objects:{object_set}"
    #model to be used for text2table
    text2table_model = 'Ollama' #options: [Ollama, GPT3, Seq2Seq]

    text2sql_model = 'OpenAI' #options: [OpenAI, DeepSeek, HuggingFace, Anthropic, HuggingFace]
    text2sql_model_name = 'gpt3.5-turbo'

    #models to be used for captioning
    caption_model = 'LLamaVision'


    #only for debugging
    debug = True

    # batch size for inserting into the DB
    batch_size = 40
    frames_per_video = 10

    #schema definition for databases
    caption_schema = dict(video_id = "INTEGER UNIQUE NOT NULL", frame_id = "INTEGER UNIQUE NOT NULL", description="TEXT NOT NULL", objects="TEXT")
    table_name = "videos"
    base_db_path = '/users/ssunda11/git/CS2270-Video-Captioning/database_integration'
    db_name = "video_frames.db"