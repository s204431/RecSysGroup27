from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import time
import pandas as pd
import Utils
from datetime import datetime

precompute_token_dicts = True

def flatten(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):  # Hvis elementet er en liste
            flattened_list.extend(flatten(item))  # Rekursiv kald for at flade den liste
        else:
            flattened_list.append(item)  # Hvis elementet ikke er en liste, tilføj direkte
    return flattened_list

def find_position(row):
    clicked_value = row['article_ids_clicked'][0] 
    inview_list = row['article_ids_inview']  
    if isinstance(inview_list, np.ndarray):  
        inview_list = inview_list.tolist() 
    if clicked_value in inview_list:
        return int(inview_list.index(clicked_value)) 
    return None

def replace_ids_with_titles(article_ids, article_dict, subtitle_dict):
    return [f"{article_dict.get(article_id, '')} {subtitle_dict.get(article_id, '')}" for article_id in article_ids]

def encode_history_times(time_strings):
    january_first_2023 = datetime(2023, 1, 1)
    return [(datetime.fromisoformat(str(time_string)) - january_first_2023).total_seconds() / (60*60*24) for time_string in time_strings]

def encode_impression_time(time_string):
    january_first_2023 = datetime(2023, 1, 1)
    return (datetime.fromisoformat(str(time_string)) - january_first_2023).total_seconds() / (60*60*24)

class ArticlesDatasetTraining(Dataset):
    def __init__(self, DATASET, type, nlp): #Type is "train", "validation"
        start = time.time()
        PATH = Path(__file__).parent.resolve().joinpath("./ebnerd_data")
    
        df_history   = pd.read_parquet(PATH.joinpath(DATASET, type, "history.parquet"))
        df_behaviors = pd.read_parquet(PATH.joinpath(DATASET, type, "behaviors.parquet"))
        df_articles  = pd.read_parquet(PATH.joinpath(DATASET, "articles.parquet"))
        df_history = df_history[['user_id','article_id_fixed','impression_time_fixed']]
        df_articles = df_articles[['article_id', 'title', 'subtitle']]
        df_behaviors = df_behaviors[['user_id', 'article_ids_inview', 'article_ids_clicked', 'impression_time']]
        df_behaviors[['impression_time']] = df_behaviors[['impression_time']].map(encode_impression_time)
        self.df_data = df_behaviors
        self.article_dict = pd.Series(df_articles['title'].values,index=df_articles['article_id']).to_dict()
        self.subtitle_dict = pd.Series(df_articles['subtitle'].values,index=df_articles['article_id']).to_dict()
        #df_history[['article_titles_fixed']] = df_history[['article_id_fixed']].map(replace_ids_with_titles, article_dict=self.article_dict, subtitle_dict=self.subtitle_dict)
        #df_behaviors[['article_titles_inview', 'article_titles_clicked']] = df_behaviors[['article_ids_inview', 'article_ids_clicked']].map(replace_ids_with_titles, article_dict=self.article_dict, subtitle_dict=self.subtitle_dict)
        self.combined_dict = {article_id: f"{self.article_dict.get(article_id, '')} {self.subtitle_dict.get(article_id, '')}" for article_id in set(self.article_dict).union(self.subtitle_dict)}
        if precompute_token_dicts:
            self.combined_dict = {article_id: Utils.replace_titles_with_tokens([self.combined_dict.get(article_id)], nlp=nlp, max_vocab_size=len(nlp.vocab.vectors))[0] for article_id in self.combined_dict}
            mapping = lambda article_ids: [self.combined_dict.get(article_id, '') for article_id in article_ids]
        else:
            mapping = lambda article_ids: [f"{self.combined_dict.get(article_id, '')}" for article_id in article_ids]
        df_history[['article_titles_fixed']] = df_history[['article_id_fixed']].map(mapping)
        df_behaviors[['article_titles_inview', 'article_titles_clicked']] = df_behaviors[['article_ids_inview', 'article_ids_clicked']].map(mapping)
        df_history[['impression_time_fixed']] = df_history[['impression_time_fixed']].map(encode_history_times)
        self.history_dict = pd.Series(df_history['article_titles_fixed'].values,index=df_history['user_id']).to_dict()
        self.time_dict = pd.Series(df_history['impression_time_fixed'].values,index=df_history['user_id']).to_dict()
        print("Time to load data: ", time.time() - start)


    def __len__(self):
        """
        Returner længden af data (antallet af rækker i df_history).
        """
        return len(self.df_data)

    def __getitem__(self, idx):
        """
        Fetch user history and target article for a given index.
        """
        row = self.df_data.iloc[idx]
        return row['user_id'], row['article_titles_inview'], row['article_titles_clicked'], row['impression_time']

class ArticlesDatasetTest(Dataset):
    def __init__(self, DATASET, nlp):
        start = time.time()
        PATH = Path(__file__).parent.resolve().joinpath("./ebnerd_data")
    
        df_history   = pd.read_parquet(PATH.joinpath(DATASET, "test", "history.parquet"))
        df_behaviors = pd.read_parquet(PATH.joinpath(DATASET, "test", "behaviors.parquet"))
        df_articles  = pd.read_parquet(PATH.joinpath(DATASET, "articles.parquet"))
        df_history = df_history[['user_id','article_id_fixed']]
        df_articles = df_articles[['article_id', 'title', 'subtitle']]
        df_behaviors = df_behaviors[['impression_id', 'user_id', 'article_ids_inview']]
        self.df_data = df_behaviors
        self.article_dict = pd.Series(df_articles['title'].values,index=df_articles['article_id']).to_dict()
        self.subtitle_dict = pd.Series(df_articles['subtitle'].values,index=df_articles['article_id']).to_dict()
        #df_history[['article_titles_fixed']] = df_history[['article_id_fixed']].map(replace_ids_with_titles, article_dict=self.article_dict, subtitle_dict=self.subtitle_dict)
        #df_behaviors[['article_titles_inview']] = df_behaviors[['article_ids_inview']].map(replace_ids_with_titles, article_dict=self.article_dict, subtitle_dict=self.subtitle_dict)
        self.combined_dict = {article_id: f"{self.article_dict.get(article_id, '')} {self.subtitle_dict.get(article_id, '')}" for article_id in set(self.article_dict).union(self.subtitle_dict)}
        if precompute_token_dicts:
            self.combined_dict = {article_id: Utils.replace_titles_with_tokens([self.combined_dict.get(article_id)], nlp=nlp, max_vocab_size=len(nlp.vocab.vectors))[0] for article_id in self.combined_dict}
            mapping = lambda article_ids: [self.combined_dict.get(article_id, '') for article_id in article_ids]
        else:
            mapping = lambda article_ids: [f"{self.combined_dict.get(article_id, '')}" for article_id in article_ids]
        df_history[['article_titles_fixed']] = df_history[['article_id_fixed']].map(mapping)
        df_behaviors[['article_titles_inview']] = df_behaviors[['article_ids_inview']].map(mapping)
        self.history_dict = pd.Series(df_history['article_titles_fixed'].values,index=df_history['user_id']).to_dict()
        print("Time to load data: ", time.time() - start)


    def __len__(self):
        """
        Returner længden af data (antallet af rækker i df_history).
        """
        return len(self.df_data)

    def __getitem__(self, idx):
        """
        Fetch user history and target article for a given index.
        """
        row = self.df_data.iloc[idx]
        return row['impression_id'], row['user_id'], row['article_titles_inview']

'''
dataset = ArticlesDatasetTraining('ebnerd_small', 'train')
print(dataset[0])
#Random article ids: [9778623, 9778682, 9778669, 9778657, 9778736, 9778728]
print("Random Article:\n", dataset.combined_dict[9778623])
if precompute_token_dicts:
    print("Tokenized Article:\n", dataset.combined_token_dict[9778623])

test_dataset = ArticlesDatasetTest('ebnerd_testset')
print(test_dataset[0])
#Clicked article ids: [9796527, 7851321, 9798805, 9795150, 9531110, 9798526, 9798682, 9796198, 9492777]
print("Random Clicked Article:\n", test_dataset.combined_dict[7851321])
if precompute_token_dicts:
    print("Tokenized Article:\n", test_dataset.combined_token_dict[7851321])
'''