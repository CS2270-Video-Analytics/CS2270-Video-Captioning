import torch

class Config:
    """Main configuration for the video captioning system"""
    #-------------------------------------------------------------------------
    # General settings
    #-------------------------------------------------------------------------
    debug = True
    save_frames = False
    model_precision = torch.float16
    num_threads = 8

    #-------------------------------------------------------------------------
    # Captioning Pipeline settings
    #-------------------------------------------------------------------------
    caption_model = 'LLamaVision'
    vision_model_name = 'llama3.2-vision:11b'
    clip_model_name = "ViT-L/14"
    max_tokens = 300
    keep_alive = 60000
    obj_focus = True
    previous_frames = True
    system_eval = False
    sliding_window_size = 1
    frames_per_video = 40

    # Captioning pipeline prompts
    question_prompt_format = "Question: {question} Answer:"
    sliding_window_caption_prompt_format = \
        "Task: given the current driving scene frame recorded from a dash cam angle, output a chunk of descriptions per key distinct object in the format of the general template below. " \
        "object id: the object's id (start from 1, auto increment by object) " \
        "(1) category: a high level category the object belongs to, which doesn't contain any attribute level details " \
        "(2) attributes: object's key identifying attributes, formatted as a string separated by comma " \
        "(3) action: object's state or action inferred " \
        "(4) location: object's relative location to other key objects in the frame " \
        "Note: " \
        "- object's category should only consider the typical categories seen in traffic scene, such as vehicle, pedestrian, traffic light, building, road, etc " \
        "- for object's category, try to reuse category name from previously seen object categories in {object_set} " \
        "- put same category object's chunk next to each other " \
        "- keep description concise " \
        "Example: " \
        "object id: 1 " \
        "(1) category " \
        "(2) attributes " \
        "(3) action " \
        "(4) location " \
        "object id: 2 " \
        "(1) category " \
        "(2) attributes " \
        "(3) action " \
        "(4) location " \
        "... " \
        "Task (for the current frame): " \
        "object id:"

    object_extraction_prompt_format = \
        "Task: given current frame description and seen object categories so far, output a list of unique unseen new object categories from the current frame's description that are not seen before. Don't return me the code as I want the actual the output. " \
        "General template: " \
        "current frame description: " \
        "    object id: the object's id " \
        "    (1) category: a high level category the object belongs to, which doesn't contain any attribute level details " \
        "    (2) attributes: object's key identifying attributes, formatted as a string separated by comma " \
        "    (3) action: object's state or action inferred " \
        "    (4) location: object's relative location to other key objects in the frame " \
        "seen object categories: a list of unique seen object categories before " \
        "unseen objects categories: a list of unseen object categories in the current frame; for each object in the frame, only append the category to unseen list here if its significantly different from seen categories. Otherwise count as seen object categories and don't include its category here. " \
        "Example 1 (non-existent just for illustration): " \
        "current frame description: " \
        "    object id: 1 " \
        "    category: A " \
        "    attributes: some attribute " \
        "    action: some attribute " \
        "    location: some location " \
        "seen object categories: [] " \
        "unseen objects categories: [A] " \
        "Example 2 (non-existent just for illustration): " \
        "current frame description: " \
        "    object id: 1 " \
        "    category: A " \
        "    attributes: some attribute " \
        "    action: some attribute " \
        "    location: some location " \
        "    object id: 2 " \
        "    category: C " \
        "    attributes: some attributes " \
        "    action: some attributes " \
        "    location: some location " \
        "seen object categories: [A] " \
        "unseen objects categories: [C] " \
        "Task (for the current frame): " \
        "current frame description: {curr_img_caption} " \
        "seen objects categories: [{object_set}] " \
        "unseen objects categories:"

    #-------------------------------------------------------------------------
    # Text2Table Pipeline settings
    #-------------------------------------------------------------------------
    text2table_model = 'GPT'  # options: [Ollama, GPT, Seq2Seq]
    
    # Text2Table pipeline prompts
    text2table_attribute_extraction_prompt = \
    "All image captions: {incontext_captions} " \
    "Given captions of all images, generate a list of important attributes " \
    "that can describe the objects in the image: " \
    "[color, brand, size, action, speed, location, position, state] " \
    "All image captions: {frame_captions} " \
    "Given captions of all images, generate a list of important attributes " \
    "that can describe objects in the image:"

    text2table_incontext_prompt = "\n-".join([
        "A red Nissan SUV is in the center of the road, putting on the breaks, whilst a young female pedestrian wearing yellow shirt crosses", 
        "A red Nissan SUV driving to the left lane and a grey Chevrolet and blue motorbike on the right lane. The motorbike is speeding without breaks"
    ])
    
    text2table_frame_prompt = \
        "Given a detailed description of an image and the set of objects below, output a structured table with the columns {formatted_schema}. " \
        "Separate the start/end of table rows with <r> and start/end of table columns with <c> in one line. STRICTLY follow the format. " \
        "object: one of the objects in set of objects " \
        "attributes: detailed descriptive qualities about the object and its properties; where attribute is not present, leave empty " \
        "image_location: spatial location of object in the image relative to other objects " \
        "action: the detailed process or movement the object is performing; where action is not relevant for an object, leave empty " \
        "Detailed image description:A silver audi car with license plate \"ABCD-1234\", that is driving whilst facing forward in front of a traffic light and a traffic light that is red to the right of the road " \
        "Set of objects:{object_set} " \
        "Directly generate the table with no prefix or suffix:<r><c>vehicle<c>in front of traffic light, facing forwards<c>silver Audi car with license plate \"ABCD-1234\"<c>driving forward and stopping at red-light<c><r><c>traffic light<c>right of the road in front of silver car<c>red color signalling stop<c><c><r> " \
        "Detailed image description:{image_caption} " \
        "Set of objects:{object_set} " \
        "Directly generate the table with no prefix or suffix:"
    
    #-------------------------------------------------------------------------
    # Text2SQL Pipeline settings
    #-------------------------------------------------------------------------
    text2sql_model = 'OpenAI'  # options: [OpenAI, DeepSeek, HuggingFace, Anthropic]
    text2sql_model_name = 'gpt3.5-turbo'

    #-------------------------------------------------------------------------
    # Language Model settings
    #-------------------------------------------------------------------------
    text_model_name = 'llama3.2:latest'

    #-------------------------------------------------------------------------
    # Database settings
    #-------------------------------------------------------------------------
    sql_db_path = '/users/ssunda11/git/CS2270-Video-Captioning/database_integration'
    sql_db_name = "video_frames.db"
    batch_size = 4
    vec_db_path = '/users/ssunda11/git/CS2270-Video-Captioning/database_integration'
    vec_db_name = 'video_frames.index'
    
    # Table definitions
    caption_table_name = "raw_videos"
    caption_table_pk = ['video_id', 'frame_id']
    caption_table_schema = {
        'video_id': "INTEGER", 
        'frame_id': "REAL NOT NULL", 
        'description': "TEXT NOT NULL", 
        'vector_id': "INTEGER"
    }
    
    processed_table_name = "processed_video"
    processed_table_pk = ['video_id', 'frame_id', 'object']
    processed_table_schema = {
        'video_id': "INTEGER", 
        'frame_id': "REAL NOT NULL", 
        'object': "TEXT NOT NULL", 
        'image_location': "TEXT", 
        'description': "TEXT", 
        'action': "TEXT"
    }
    