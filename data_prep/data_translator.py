from deep_translator import GoogleTranslator 
import pandas as pd
import os
from tqdm import tqdm

languages = ['English', 'German', 'French', 'Italian', 'Portuguese', 'Hindi', 'Spanish']
languages = [i.lower() for i in languages]

for lang in languages:
    if lang == 'english':
        continue
    print(f'Translating to {lang}')
    data_path = f'../data/{lang}'
    for data_type in ['train', 'dev', 'test']:
        print(f'Translating {data_type} data')
        df = pd.read_csv(os.path.join(data_path, f'{data_type}.csv'))
        df['translated_answer'] = None
        for i in tqdm(range(len(df['answer']))):
            if df['answerType'][i] in ['date', 'numerical', 'entity']:
                df.loc[i, 'translated_answer'] = df['answer'][i]
            else:
                try:
                    translated = GoogleTranslator(source='english', target=lang).translate(df['answer'][i])
                    df.loc[i, 'translated_answer'] = translated
                except:
                    answer = df['answer'][i]
                    print(f'Error in translating {answer} to {lang}')
        df.to_csv(os.path.join(data_path, f'{data_type}.csv'), index=False)