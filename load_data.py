import torch
import pandas as pd
import random
import numpy as np
import math


from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from UserEncoder import UserEncoder
from torch import nn

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
        PATH = Path(__file__).parent.resolve().joinpath("./ebnerd_data")
    
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
        self.df_data['article_ids_fixed'] = self.df_data['article_ids_fixed'].apply(replace_ids_with_titles)


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
        return row['article_ids_fixed'], row['article_ids_inview'], row['article_ids_clicked'], row['gt_position']

def getLastN(lst, N):
    if len(lst) < N:
        return lst
    else:
        return lst[-N:]

def accuracy(outputs, targets):
    nCorrect = 0
    for i in range(0, len(outputs)):
        pred = torch.argmax(outputs[i])
        if pred == targets[i]:
            nCorrect += 1
    return nCorrect/len(targets)

dataset = ArticlesDatasetTraining('ebnerd_demo', K=1)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=list)
user_encoder = UserEncoder(h=16, dropout=0.2)

optimizer = torch.optim.Adam(user_encoder.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
num_epochs = 1
user_encoder.train()
for i in range(0, num_epochs):
    accuracies = []
    for batch in train_loader:
        batch_outputs = []
        batch_targets = []
        for history, sample, target, gt_position in batch:
            if math.isnan(gt_position):
                continue
            history = getLastN(history, 10)
            print(len(history), len(sample), len(target), gt_position)
            output = user_encoder(history=history, targets=sample)
            batch_outputs.append(output)
            batch_targets.append(torch.tensor(int(gt_position)))
        loss = criterion(torch.stack(batch_outputs), torch.stack(batch_targets))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = accuracy(batch_outputs, batch_targets)
        accuracies.append(acc)
        print("Loss: ", float(loss.data.numpy()))
        print("Accuracy: ", acc)
        print("Average accuracy so far: ", sum(accuracies)/len(accuracies))
    break
