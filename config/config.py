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
    system_eval = False

    #-------------------------------------------------------------------------
    # Captioning Pipeline settings
    #-------------------------------------------------------------------------
    caption_model_name = 'OpenAI;gpt-4o-mini' #'OllamaVision;llama3.2-vision:11b' #options: [OllamaVision, OpenAI, BLIP2, BLIP] 'BLIP2/Salesforce/blip2-opt-2.7b'
    
    caption_model_params = {
        'temperature': 0.7,
        'top_k': 3,
        'top_p': 0.9,
        'num_ctx': 2048,
        'repeat_penalty': 0.5,
        'presence_penalty': 0.7,
        'frequency_penalty':0.3,
        'max_tokens': 2048,
        'stop_tokens': None,
        'keep_alive': 30000
    }
    caption_detail = 'high'
    obj_focus = True
    init_object_set = []
    previous_frames = True
    sliding_window_size = 1
    frames_per_video = 40
    clip_model_name = "ViT-L/14"

    # Captioning pipeline prompts
    captioning_context_prompt_format = \
    "You are a system designed to provide detailed captions for images (frames) in a video"
    question_prompt_format = "Question: {question} Answer:"
    sliding_window_caption_prompt_format = \
        "Task: given the current scene frame recorded from a camera angle, output a chunk of descriptions per key distinct object in the format of the general template below. " \
        "object id: the object's id (start from 1, auto increment by object) " \
        "(1) category: a high level category the object belongs to, which doesn't contain any attribute level details " \
        "(2) attributes: object's key identifying attributes, formatted as a string separated by comma " \
        "(3) action: object's state or action inferred " \
        "(4) location: object's relative location to other key objects in the frame " \
        "Note: " \
        "- object's category should only consider the typical categories seen in the current scene" \
        "- for object's category, try to reuse category name from previously seen object categories in {object_set} " \
        "- put same category object's chunk next to each other " \
        "- capture key identifying attributes per object so we can differentiate between objects of the same category. \
            For example for person, gender, height, clothing, race etc are important. For vehicle brand, model, color, license, etc are important. Apply the similar idea to other categories." \
        "- However, for attributes, please only include those that we can identify well. If not confident, don't include it as we want included information to be accurate." \
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

    prior_object_extraction_prompt = "Task: given the current frame, extract a list of object categories visible in the frame\nobjects:["
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
        "unseen objects categories: a list of objects in the current frame description that are NOT in seen object categories " \
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
        "current frame description: {curr_img_caption}\n\n" \
        "seen objects categories: [{object_set}] " \
        "unseen objects categories:"

    #-------------------------------------------------------------------------
    # Text2Table Pipeline settings
    #-------------------------------------------------------------------------
    db_path = './database_integration/video_frames.db'
    text2table_model_name = 'OpenAI;gpt-4o-mini'  # options: [Ollama, OpenAI, Seq2Seq]
    text2table_params = {
        'temperature': 0.5,
        'top_k': None,
        'top_p': None,
        'num_ctx': None,
        'repeat_penalty': None,
        'presence_penalty': None,
        'frequency_penalty':None,
        'max_tokens': None,
        'stop_tokens': None,
        'keep_alive': None,
        'batch_size': None,
        'num_threads': None,
        'model_precision': None,
        'system_eval': False,
    }

    #Text2Table SQL queries
    combined_description_query =\
    """
        SELECT GROUP_CONCAT(description, ' ') AS combined_description
        FROM (
            SELECT description FROM {caption_table_name} LIMIT ?
        )
    """

    #Text2Table pipeline prompts
    text2table_incontext_prompt = \
    """
    Sample text :
    < tr class =" mergedrow " > < th scope =" row " class =" infobox -
    label " > < div style =" text - indent : -0.9 em ; margin - left
    :1.2 em ; font - weight : normal ;" > < a href ="/ wiki /
    Monarchy_of_Canada " title =" Monarchy of Canada " >
    Monarch </ a > </ div > </ th > < td class =" infobox - data " > <
    a href ="/ wiki / Charles_III " title =" Charles III " >
    Charles III </ a > </ td > </ tr >
    < tr class =" mergedrow " > < th scope =" row " class =" infobox -
    label " > < div style =" text - indent : -0.9 em ; margin - left
    :1.2 em ; font - weight : normal ;" > < span class =" nowrap
    " > < a href ="/ wiki / Governor_General_of_Canada "
    title =" Governor General of Canada " > Governor
    General </ a > </ span > </ div > </ th > < td class =" infobox -
    data " > < a href ="/ wiki / Mary_Simon " title =" Mary
    Simon " > Mary Simon </ a > </ td > </ tr >
    <b > Provinces and Territories </ b class =' navlinking
    countries '>
    <ul >
    <li > Saskatchewan </ li >
    <li > Manitoba </ li >
    <li > Ontario </ li >
    <li > Quebec </ li >
    <li > New Brunswick </ li >
    <li > Prince Edward Island </ li >
    <li > Nova Scotia </ li >
    <li > Newfoundland and Labrador </ li >
    <li > Yukon </ li >
    <li > Nunavut </ li >
    <li > Northwest Territories </ li >
    </ ul >
    Question : List all relevant attributes about 'Canada ' that are exactly mentioned in this sample text if
    any .
    Answer :
    - Monarch : Charles III
    - Governor General : Mary Simon
    - Provinces and Territories : Saskatchewan , Manitoba ,
    Ontario , Quebec , New Brunswick , Prince Edward
    Island , Nova Scotia , Newfoundland and Labrador ,
    Yukon , Nunavut , Northwest Territories
    ----
    Sample text :
    Patient birth date : 1990 -01 -01
    Prescribed medication : aspirin , ibuprofen ,
    acetaminophen
    Prescribed dosage : 1 tablet , 2 tablets , 3 tablets
    Doctor 's name : Dr . Burns
    Date of discharge : 2020 -01 -01
    Hospital address : 123 Main Street , New York , NY 10001
    Question : List all relevant attributes about 'medications ' that are exactly mentioned in this sample text if any .
    Answer :
    - Prescribed medication : aspirin , ibuprofen , acetaminophen
    - Prescribed dosage : 1 tablet , 2 tablets , 3 tablets
    ----
    """
    text2table_attribute_extraction_prompt =\
    """
    {incontext_examples}
    ----
    Sample text :
    {all_joined_captions}
    Question : List all unique categories and category associated attributes that are exactly mentioned in this sample
    text in the format of category: attributes for each category
    Answer :
    """

    text2table_schema_generation_prompt =\
    """
    Given the extracted categories and attributes for each object in the current video, generate SQL CREATE TABLE statements for each unique category that does not already have a table. 
    Each table must include the following columns:
    - "video_id" (TEXT, not null)
    - "frame_id" (REAL, not null)
    - "object_id" (INTEGER)
    - "location" (TEXT)
    
    Addtionally, 
    (1) for living categories that can perform actions (e.g., person, animal, etc), include another "action" column in the corresponding table. Otherwise don't.
    (2) for each category's attributes we should create meaningful columns that are representatitve and extractable. For example, for a person with attribute description "male wearing black shirt, holding papers", creating attribute columns like gender, clothing, attachment, action would make sense. For objects like furniture, type, color, design, material, etc would make sense.
    (3) avoid using reserved SQL keywords as table name and if necessary, append '_data' to the category name to form the table name to avoid error.
    (4) consider carefully if an attribute can be part of table's column or a new table, pick one that's most relevant to the object.
    (5) for output, generate only the sql create table statements and don't include anything else before or after.

    Extracted category-attribute pairs:
    {attributes}

    Existing tables for previously seen categories:
    {existing_tables}

    Generate only the SQL CREATE TABLE statements for all new tables:
    """

    text2table_frame_prompt = """
    You are an expert at converting natural language descriptions of {scene_descriptor} into structured data.

    Given the following description:

    "{description}"

    Extract the structured information and return it as JSON using this format:

    {{
    {json_schema}
    }}

    Only include fields that can be reasonably inferred. Leave out objects that are not mentioned. Use null for missing but relevant fields.
    """

    # Text2Table pipeline prompts
    # text2table_attribute_extraction_prompt = \
    # "All image captions: {incontext_captions} " \
    # "Given captions of all images, generate a list of important attributes " \
    # "that can describe the objects in the image: " \
    # "[color, brand, size, action, speed, location, position, state] " \
    # "All image captions: {frame_captions} " \
    # "Given captions of all images, generate a list of important attributes " \
    # "that can describe objects in the image:"

    # text2table_incontext_prompt = "\n-".join([
    #     "A red Nissan SUV is in the center of the road, putting on the breaks, whilst a young female pedestrian wearing yellow shirt crosses", 
    #     "A red Nissan SUV driving to the left lane and a grey Chevrolet and blue motorbike on the right lane. The motorbike is speeding without breaks"
    # ])
    
    # text2table_frame_prompt = \
    #     "Given a detailed description of an image and the set of objects below, output a structured table with the columns {formatted_schema}. " \
    #     "Separate the start/end of table rows with <r> and start/end of table columns with <c> in one line. STRICTLY follow the format. " \
    #     "object: one of the objects in set of objects " \
    #     "attributes: detailed descriptive qualities about the object and its properties; where attribute is not present, leave empty " \
    #     "image_location: spatial location of object in the image relative to other objects " \
    #     "action: the detailed process or movement the object is performing; where action is not relevant for an object, leave empty " \
    #     "Detailed image description:A silver audi car with license plate \"ABCD-1234\", that is driving whilst facing forward in front of a traffic light and a traffic light that is red to the right of the road " \
    #     "Set of objects:{object_set} " \
    #     "Directly generate the table with no prefix or suffix:<r><c>vehicle<c>in front of traffic light, facing forwards<c>silver Audi car with license plate \"ABCD-1234\"<c>driving forward and stopping at red-light<c><r><c>traffic light<c>right of the road in front of silver car<c>red color signalling stop<c><c><r> " \
    #     "Detailed image description:{caption} " \
    #     "Set of objects:{object_set} " \
    #     "Directly generate the table with no prefix or suffix:"
    
    #-------------------------------------------------------------------------
    # Text2SQL Pipeline settings
    #-------------------------------------------------------------------------
    text2sql_model_name = 'OpenAI;gpt-4o-mini' # options: [OpenAI;, DeepSeek;deepseek-chat, HuggingFace;, Anthropic;claude-3-5-haiku-latest]
    text2sql_params = {
        'temperature': 1,
        'top_k': 3,
        'top_p': 1,
        'num_ctx': 2048,
        'repeat_penalty': 0.5,
        'presence_penalty': 0.0,
        'frequency_penalty':0.0,
        'max_tokens': 200,
        'stop_tokens': None,
        'keep_alive': 30000
    }

    # Important SQLite Constraints
    #     - Use **only INNER JOIN, LEFT JOIN, or CROSS JOIN** (no FULL OUTER JOIN).
    #     - **No native JSON functions** (assume basic text handling).
    #     - Data types are flexible; prefer **TEXT, INTEGER, REAL, and BLOB**.
    #     - **BOOLEAN is represented as INTEGER** (0 = False, 1 = True).
    #     - Use **LOWER()** for case-insensitive string matching.
    #     - Primary keys auto-increment without `AUTOINCREMENT` unless explicitly required.
    #     - Always assume **foreign key constraints are disabled unless explicitly turned on**.

    @staticmethod
    def get_text2sql_prompt(schema_info: str, question: str) -> str:
        return f"""
        You are an AI thay converts user's natural language question into a SQL query with the context of 
        existing table schemas below from a sqlite database. Based on user's question and existing tables' schemas
        below, select all possible relevant tables and relevant attributes to the query to construct the SQL query. 
        
        Note that:
        1. for the SQL query, only output a valid query that's executable without any additional text like prefix, suffix or formatting like ```sql, ```.
        2. for each table selected, only use existing columns and the column to use should make sense
        3. can join relevant tables when necessary since video_id and frame_id are unique
        4. if possible duplicate, deduplicate
        5. try using like instead of = for filtering
        6. if multiple columns may contain the searching attributes and you are not sure, try use OR instead of AND for filtering

        Database table schemas:
        {schema_info}

        User question:
        {question}

        SQL query:
        """

    #-------------------------------------------------------------------------
    # Database settings
    #-------------------------------------------------------------------------
    sql_db_path = './database_integration'
    sql_db_name = "video_frames.db"
    batch_size = 4
    vec_db_path = './database_integration'
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
    