from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import time
import pandas as pd
import Utils
from datetime import datetime

precompute_token_dicts = True

def flatten(lst):
    """Recursively flattens a nested list into a single-level list"""
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list

def find_position(row):
    """Finds the position of the first clicked article in the inview list"""
    clicked_value = row['article_ids_clicked'][0] 
    inview_list = row['article_ids_inview']  
    if isinstance(inview_list, np.ndarray):  
        inview_list = inview_list.tolist() 
    if clicked_value in inview_list:
        return int(inview_list.index(clicked_value)) 
    return None

def replace_ids_with_titles(article_ids, article_dict, subtitle_dict):
    """Replaces article IDs with their corresponding titles and subtitles"""
    return [f"{article_dict.get(article_id, '')} {subtitle_dict.get(article_id, '')}" for article_id in article_ids]

def encode_history_times(time_strings):
    """Encodes a list of history times into hours since January 1, 1970"""
    january_first_1970 = datetime(1970, 1, 1)
    return [(datetime.fromisoformat(str(time_string)) - january_first_1970).total_seconds() / (60*60) for time_string in time_strings]

def encode_impression_time(time_string):
    """Encodes an impression time into hours since January 1, 1970"""
    january_first_1970 = datetime(1970, 1, 1)
    return (datetime.fromisoformat(str(time_string)) - january_first_1970).total_seconds() / (60*60)

def encode_article_times(article_ids, dict):
    """Encodes article times into hours since January 1, 1970"""
    january_first_1970 = datetime(1970, 1, 1)
    return [(datetime.fromisoformat(str(dict[id])) - january_first_1970).total_seconds() / (60*60) for id in article_ids]

def remove_duplicates_until_n_left(data_frame, n):
    """Removes dupliates of clicked articles until only n duplicates are left"""
    #df_behaviors = df_behaviors[~df_behaviors.duplicated(subset='article_ids_clicked', keep='first')]
    seen_entries = {}

    def filter_func(v):
        key = str(v)
        value = seen_entries.get(key, 1)
        if value > n:
            return False
        seen_entries[key] = value + 1
        return True
    
    return data_frame[data_frame['article_ids_clicked'].apply(filter_func)]

class ArticlesDatasetTraining(Dataset):
    def __init__(self, DATASET, type, nlp): #Type is "train", "validation"
        start = time.time()
        PATH = Path(__file__).parent.resolve().joinpath("./ebnerd_data")
    
        # Load parquet files for user history, behaviors, and articles
        df_history   = pd.read_parquet(PATH.joinpath(DATASET, type, "history.parquet"))
        df_behaviors = pd.read_parquet(PATH.joinpath(DATASET, type, "behaviors.parquet"))
        df_articles  = pd.read_parquet(PATH.joinpath(DATASET, "articles.parquet"))

        # Only keep relevant columns from dataframes
        df_history = df_history[['user_id','article_id_fixed','impression_time_fixed']]
        df_articles = df_articles[['article_id', 'title', 'subtitle', 'published_time']]
        df_behaviors = df_behaviors[['user_id', 'article_ids_inview', 'article_ids_clicked', 'impression_time']]

        # Transform the data and create useful dictionaries (for performance)
        #df_behaviors = remove_duplicates_until_n_left(df_behaviors, 10)
        df_behaviors[['impression_time']] = df_behaviors[['impression_time']].map(encode_impression_time)
        self.df_data = df_behaviors
        self.article_dict = pd.Series(df_articles['title'].values,index=df_articles['article_id']).to_dict()
        self.subtitle_dict = pd.Series(df_articles['subtitle'].values,index=df_articles['article_id']).to_dict()
        self.article_time_dict = pd.Series(df_articles['published_time'].values,index=df_articles['article_id']).to_dict()
        self.combined_dict = {article_id: f"{self.article_dict.get(article_id, '')} {self.subtitle_dict.get(article_id, '')}" for article_id in set(self.article_dict).union(self.subtitle_dict)}
        if precompute_token_dicts:
            self.combined_dict = {article_id: Utils.replace_titles_with_tokens([self.combined_dict.get(article_id)], nlp=nlp, max_vocab_size=len(nlp.vocab.vectors))[0] for article_id in self.combined_dict}
            mapping = lambda article_ids: [self.combined_dict.get(article_id, '') for article_id in article_ids]
        else:
            mapping = lambda article_ids: [f"{self.combined_dict.get(article_id, '')}" for article_id in article_ids]
        df_history[['article_titles_fixed']] = df_history[['article_id_fixed']].map(mapping)
        df_history[['published_time_fixed']] = df_history[['article_id_fixed']].map(lambda ids: encode_article_times(ids, self.article_time_dict))
        df_behaviors[['article_titles_inview', 'article_titles_clicked']] = df_behaviors[['article_ids_inview', 'article_ids_clicked']].map(mapping)
        df_behaviors[['article_times_inview', 'article_times_clicked']] = df_behaviors[['article_ids_inview', 'article_ids_clicked']].map(lambda ids: encode_article_times(ids, self.article_time_dict))
        self.history_dict = pd.Series(df_history['article_titles_fixed'].values,index=df_history['user_id']).to_dict()
        self.time_dict = pd.Series(df_history['published_time_fixed'].values,index=df_history['user_id']).to_dict()
        print("Time to load data: ", time.time() - start)


    def __len__(self):
        """
        Return the length of the data (number of rows in df_history).
        """
        return len(self.df_data)

    def __getitem__(self, idx):
        """
        Fetch user history, target article, imo. for a given index.
        """
        row = self.df_data.iloc[idx]
        return row['user_id'], row['article_titles_inview'], row['article_times_inview'], row['impression_time'], row['article_titles_clicked'], row['article_times_clicked']

class ArticlesDatasetTest(Dataset):
    def __init__(self, DATASET, nlp):
        start = time.time()
        PATH = Path(__file__).parent.resolve().joinpath("./ebnerd_data")
    
        # Load parquet files for user history, behaviors, and articles
        df_history   = pd.read_parquet(PATH.joinpath(DATASET, "test", "history.parquet"))
        df_behaviors = pd.read_parquet(PATH.joinpath(DATASET, "test", "behaviors.parquet"))
        df_articles  = pd.read_parquet(PATH.joinpath(DATASET, "articles.parquet"))

        # Only keep relevant columns from dataframes
        df_history = df_history[['user_id','article_id_fixed','impression_time_fixed']]
        df_articles = df_articles[['article_id', 'title', 'subtitle', 'published_time']]
        df_behaviors = df_behaviors[['impression_id', 'user_id', 'article_ids_inview','impression_time']]

        # Transform the data and create useful dictionaries (for performance)
        df_behaviors[['impression_time']] = df_behaviors[['impression_time']].map(encode_impression_time)
        self.df_data = df_behaviors
        self.article_dict = pd.Series(df_articles['title'].values,index=df_articles['article_id']).to_dict()
        self.subtitle_dict = pd.Series(df_articles['subtitle'].values,index=df_articles['article_id']).to_dict()
        self.article_time_dict = pd.Series(df_articles['published_time'].values,index=df_articles['article_id']).to_dict()
        self.combined_dict = {article_id: f"{self.article_dict.get(article_id, '')} {self.subtitle_dict.get(article_id, '')}" for article_id in set(self.article_dict).union(self.subtitle_dict)}
        if precompute_token_dicts:
            self.combined_dict = {article_id: Utils.replace_titles_with_tokens([self.combined_dict.get(article_id)], nlp=nlp, max_vocab_size=len(nlp.vocab.vectors))[0] for article_id in self.combined_dict}
            mapping = lambda article_ids: [self.combined_dict.get(article_id, '') for article_id in article_ids]
        else:
            mapping = lambda article_ids: [f"{self.combined_dict.get(article_id, '')}" for article_id in article_ids]
        df_history[['article_titles_fixed']] = df_history[['article_id_fixed']].map(mapping)
        #df_history[['impression_time_fixed']] = df_history[['impression_time_fixed']].map(encode_history_times)
        df_history[['published_time_fixed']] = df_history[['article_id_fixed']].map(lambda ids: encode_article_times(ids, self.article_time_dict))
        df_behaviors[['article_titles_inview']] = df_behaviors[['article_ids_inview']].map(mapping)
        df_behaviors[['article_times_inview']] = df_behaviors[['article_ids_inview']].map(lambda ids: encode_article_times(ids, self.article_time_dict))
        self.history_dict = pd.Series(df_history['article_titles_fixed'].values,index=df_history['user_id']).to_dict()
        self.time_dict = pd.Series(df_history['published_time_fixed'].values,index=df_history['user_id']).to_dict()
        print("Time to load data: ", time.time() - start)


    def __len__(self):
        """
        Return the length of the data (number of rows in df_history).
        """
        return len(self.df_data)

    def __getitem__(self, idx):
        """
        Fetch user history and target article for a given index.
        """
        row = self.df_data.iloc[idx]
        return row['impression_id'], row['user_id'], row['article_titles_inview'], row['article_times_inview'], row['impression_time']