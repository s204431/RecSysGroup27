from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import time
import pandas as pd

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

class ArticlesDatasetTraining(Dataset):
    def __init__(self, DATASET, type): #Type is "train", "validation"
        start = time.time()
        PATH = Path(__file__).parent.resolve().joinpath("./ebnerd_data")
    
        df_history   = pd.read_parquet(PATH.joinpath(DATASET, type, "history.parquet"))
        df_behaviors = pd.read_parquet(PATH.joinpath(DATASET, type, "behaviors.parquet"))
        df_articles  = pd.read_parquet(PATH.joinpath(DATASET, "articles.parquet"))
        df_history = df_history[['user_id','article_id_fixed']]
        df_articles = df_articles[['article_id', 'title']]
        df_behaviors = df_behaviors[['user_id', 'article_ids_inview', 'article_ids_clicked']]
        self.df_data = df_behaviors
        self.article_dict = pd.Series(df_articles['title'].values,index=df_articles['article_id']).to_dict()
        self.history_dict = pd.Series(df_history['article_id_fixed'].values,index=df_history['user_id']).to_dict()
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
        return row['user_id'], row['article_ids_inview'], row['article_ids_clicked']

class ArticlesDatasetTest(Dataset):
    def __init__(self, DATASET):
        start = time.time()
        PATH = Path(__file__).parent.resolve().joinpath("./ebnerd_data")
    
        df_history   = pd.read_parquet(PATH.joinpath(DATASET, "test", "history.parquet"))
        df_behaviors = pd.read_parquet(PATH.joinpath(DATASET, "test", "behaviors.parquet"))
        df_articles  = pd.read_parquet(PATH.joinpath(DATASET, "articles.parquet"))
        df_history = df_history[['user_id','article_id_fixed']]
        df_articles = df_articles[['article_id', 'title']]
        df_behaviors = df_behaviors[['user_id', 'article_ids_inview']]
        self.df_data = df_behaviors
        self.article_dict = pd.Series(df_articles['title'].values,index=df_articles['article_id']).to_dict()
        self.history_dict = pd.Series(df_history['article_id_fixed'].values,index=df_history['user_id']).to_dict()
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
        return row['user_id'], row['article_ids_inview']