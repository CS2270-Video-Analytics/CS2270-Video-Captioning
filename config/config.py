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
    text2table_model_name = 'OpenAI;gpt-4o-mini'  # options: [Ollama, OpenAI, Seq2Seq]
    text2table_params = {
        'temperature': 0.2,
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
    Question : List all relevant attributes about '{scene_descriptor} ' that are exactly mentioned in this sample
    text if any .
    Answer :
    """

    text2table_schema_generation_prompt =\
    """
    Based on the following attributes, design a Database schema for a Sqlite3 Database with the
    attribute names as column headers.

    Each table you create MUST have a column named "frame_id" which is SHOULD NOT be the PRIMARY KEY and SHOULD BE of TYPE REAL.

    ONLY RETURN THE SQLITE3 TABLE CREATION STATEMENT FOR ALL THE TABLES
    DO NOT RETURN ANY OTHER TEXT

    Attributes:
    {attributes}
    Database schema:
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

    text2table_scene_description_prompt = """
    You are given a list of object-level captions describing elements detected in a video frame. 
    Based on these descriptions, summarize the overall scene depicted in the video using a short phrase of 1–3 words. 
    The summary should capture the general setting or type of scene shown (e.g., "general street scene", "kitchen interior", "sports match", "forest trail", "battlefield", "concert stage").

    Captions:
    {all_captions}

    Scene Summary:
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
        'temperature': 0,
        'top_k': None,
        'top_p': None,
        'num_ctx': None,
        'repeat_penalty': None,
        'presence_penalty': None,
        'frequency_penalty': None,
        'max_tokens': None,
        'stop_tokens': None,
        'keep_alive': None
    }

    @staticmethod
    def get_text2sql_prompt(schema_info: str, question: str) -> str:
        """Generate the prompt for text2sql model with given schema and question."""
        return f"""
        You are an AI that converts natural language questions into SQL queries
        specifically for an **SQLite3** database.

        ### **Important SQLite Constraints:**
        - Use **only INNER JOIN, LEFT JOIN, or CROSS JOIN** (no FULL OUTER JOIN).
        - **No native JSON functions** (assume basic text handling).
        - Data types are flexible; prefer **TEXT, INTEGER, REAL, and BLOB**.
        - **BOOLEAN is represented as INTEGER** (0 = False, 1 = True).
        - Use **LOWER()** for case-insensitive string matching.
        - Primary keys auto-increment without `AUTOINCREMENT` unless explicitly required.
        - Always assume **foreign key constraints are disabled unless explicitly turned on**.

        ### **Database Schema:**
        {schema_info}

        ### **User Question:**
        {question}

        ### **Expected Output:**
        Provide a valid **SQLite3-compatible** SQL query based on the question and schema.
        SQL Query:
        """

    #-------------------------------------------------------------------------
    # LLM-Judge settings
    #-------------------------------------------------------------------------

    schema_sufficiency_prompt = """
    You are a careful and knowledgeable database assistant.

    Your task is to determine whether the given SQLite3 database schema contains enough information to answer a user's question.

    ---

    ### Guidelines:

    1. **Read the question carefully.** Identify the entities, filters, or relationships needed to answer it.
    2. **Examine the schema.** Check whether the required columns, tables, and relationships are present.
    3. **Decide:**
    - If all necessary information exists in the schema, respond with:
        ```
        Sufficient: Yes
        ```
    - If anything is missing — e.g., a column, table, or relationship — respond with:
        ```
        Sufficient: No
        Missing Information:
        - <Clearly list each missing element>
        ```

    Only return the required structured response. Do **not** explain or generate SQL.

    ---

    ### User Question:
    {question}

    ---

    ### Database Schema:
    {schema_info}

    ---

    ### Your Response:
    """

    #-------------------------------------------------------------------------
    # Database settings
    #-------------------------------------------------------------------------
    # sql_db_path = '/users/ssunda11/git/CS2270-Video-Captioning/database_integration'
    sql_db_path = '/Users/pradyut/CS2270/CS2270-Video-Captioning/database_integration'
    sql_db_name = "video_frames.db"
    batch_size = 4
    # vec_db_path = '/users/ssunda11/git/CS2270-Video-Captioning/database_integration'
    vec_db_path = '/Users/pradyut/CS2270/CS2270-Video-Captioning/database_integration'
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
    