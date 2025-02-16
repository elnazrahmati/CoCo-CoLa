import json
import pandas as pd
import os



languages = ['en', 'ar', 'de', 'es', 'fr', 'it', 'ja', 'hi', 'pt']
language_abbr_map = {
    'en': 'english',
    'ar': 'arabic',
    'de': 'german',
    'es': 'spanish',
    'fr': 'french',
    'it': 'italian',
    'ja': 'japanese',
    'hi': 'hindi',
    'pt': 'portuguese'
}


for data_type in ['train', 'dev', 'test']:

    with open(os.path.join('./data', f'mintaka_{data_type}.json')) as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)

    for lang in languages:
        temp_df = pd.DataFrame()
        temp_df['id'] = df['id']
        if lang == 'en':
            temp_df['question'] = df['question']
        else:
            temp_df['question'] = [q[lang] for q in df['translations']]
        # temp_df['answer'] = [a['answer'][0]['label'][lang] for a in df['answer']]
        # if answerType is entity do above else do a['answer'][0]['label'], save the answerType in a separate column
        # temp_df['answer'] = [a['answer'][0]['label'][lang] if a['answerType'] == 'entity' else a['answer'][0] for a in df['answer']]
        answer = []
        for a in df['answer']:
            if a['answer'] is None:
                answer.append(None)
                continue
            if a['answerType'] == 'entity':
                answer.append(a['answer'][0]['label'][lang])
            else:
                answer.append(a['answer'][0])
                
        temp_df['answer'] = answer
        temp_df['answerType'] = [a['answerType'] for a in df['answer']]
        temp_df['category'] = df['category']
        temp_df['complexityType'] = df['complexityType']
        
        # create the directory if it does not exist
        if not os.path.exists(os.path.join('./data', language_abbr_map[lang])):
            os.makedirs(os.path.join('./data', language_abbr_map[lang]))
            
        temp_df.to_csv(os.path.join('./data', language_abbr_map[lang], f'{data_type}.csv'), index=False)