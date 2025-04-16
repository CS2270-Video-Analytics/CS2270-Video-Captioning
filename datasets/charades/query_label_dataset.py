import pandas as pd

charades_data = pd.read_csv('datasets/charades/Charades_v1_test_eval_10_videos.csv')
mapping_data = pd.read_csv('datasets/charades/Charades_v1_mapping.txt', sep=' ', header=None, names=['action_id', 'object_id', 'verb_id'])
verb_classes = pd.read_csv('datasets/charades/Charades_v1_verbclasses.txt', sep=' ', header=None, names=['verb_id', 'verb'])

# Create dictionaries for mapping
action_to_verb = dict(zip(mapping_data['action_id'], mapping_data['verb_id']))
verb_id_to_verb = dict(zip(verb_classes['verb_id'], verb_classes['verb']))

results = []
cleaned_data = []

for index, row in charades_data.iterrows():
    video_id = row['id']
    actions = row['actions'].split(';')
    
    # Generate video_id, verb, and verb range
    for action in actions:
        action_id, start, end = action.split()
        verb_id = action_to_verb.get(action_id)
        verb = verb_id_to_verb.get(verb_id)
        if verb:
            cleaned_data.append({
                'video_id': video_id,
                'verb': verb,
                'start_frame': float(start),
                'end_frame': float(end)
            })

# Convert to DataFrame for easy handling
cleaned_df = pd.DataFrame(cleaned_data)
cleaned_df = cleaned_df.drop_duplicates()
grouped_df = cleaned_df.groupby(['video_id', 'verb']).apply(
    lambda x: list(zip(x['start_frame'], x['end_frame']))
).reset_index(name='ranges')
# Display the cleaned data
print(grouped_df)
grouped_df.to_csv('./verb_label_range.csv', index=False)