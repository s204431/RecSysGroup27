import torch
import pandas as pd
import random
import numpy as np


from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from UserEncoder import UserEncoder

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
    def __init__(self, DATASET, K = 4):
        PATH = Path(__file__).parent.resolve().joinpath("../ebnerd_data")
    
        df_history   = pd.read_parquet(PATH.joinpath(DATASET, "train", "history.parquet"))
        df_behaviors = pd.read_parquet(PATH.joinpath(DATASET, "train", "behaviors.parquet"))
        df_articles  = pd.read_parquet(PATH.joinpath(DATASET, "articles.parquet"))
        self.df_data = df_behaviors[['user_id', 'article_ids_inview', 'article_ids_clicked']].copy()
        

        article_ids_fixed_list = []
        for index, row in self.df_data.iterrows():
            user_id = row['user_id']
            article_ids_fixed_for_user = df_history[df_history['user_id'] == user_id]['article_id_fixed'].tolist()
            article_ids_fixed_list.append(article_ids_fixed_for_user)

        self.df_data['article_ids_fixed'] = article_ids_fixed_list
        self.df_data['article_ids_fixed'] = self.df_data['article_ids_fixed'].apply(lambda x: np.concatenate(x).tolist() if isinstance(x, list) else x)

        for index, row in self.df_data.iterrows():
            article_ids_fixed = set(row['article_ids_fixed'])  
            article_ids_clicked = set(row['article_ids_clicked'])  
            remaining_articles = article_ids_fixed.difference(article_ids_clicked)
            self.df_data.at[index, 'article_ids_fixed'] = list(remaining_articles)

        for index, row in self.df_data.iterrows():
            article_ids_inview = row['article_ids_inview']
            article_ids_clicked = row['article_ids_clicked']

            if not isinstance(article_ids_clicked, list):
                if isinstance(article_ids_clicked, np.ndarray):
                    article_ids_clicked = article_ids_clicked.tolist()
                else:
                    article_ids_clicked = [article_ids_clicked]

            if isinstance(article_ids_clicked, list) and len(article_ids_clicked) > 1:
                article_ids_clicked = random.choice(article_ids_clicked)

            if isinstance(article_ids_inview, np.ndarray):
                article_ids_inview = article_ids_inview.tolist()

            if article_ids_clicked in article_ids_inview:
                article_ids_inview.remove(article_ids_clicked)

            if len(article_ids_inview) < K:
                assert len(article_ids_inview) >= K, f"Warning: There are only {len(article_ids_inview)} articles available for user_id {row['user_id']}. Cannot sample K articles."

            random_selected = random.sample(article_ids_inview, K)
            random_selected.append(article_ids_clicked)

            flattened_list = flatten(random_selected)
            random.shuffle(flattened_list)
            self.df_data.at[index, 'article_ids_inview'] = flattened_list

        self.df_data['gt_position'] = self.df_data.apply(find_position, axis=1)
        self.article_dict = pd.Series(df_articles['title'].values, index=df_articles['article_id']).to_dict()

        def replace_ids_with_titles(article_ids):
            return [self.article_dict.get(article_id, article_id) for article_id in article_ids]
        
        self.df_data['article_ids_inview'] = self.df_data['article_ids_inview'].apply(replace_ids_with_titles)
        self.df_data['article_ids_clicked'] = self.df_data['article_ids_clicked'].apply(replace_ids_with_titles)
        self.df_data['article_ids_inview'] = self.df_data['article_ids_inview'].apply(replace_ids_with_titles)


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
        return row['article_ids_inview'], row['article_ids_clicked'], row['gt_position']

dataset = ArticlesDatasetTraining('ebnerd_demo')
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=list)
user_encoder = UserEncoder(h=16, dropout=0.2)

num_epochs = 1
for batch in train_loader:
    for history, targets, gt_position in batch:
        print(history)
        print(targets)
        output = user_encoder(history=history, targets=targets)
        break
