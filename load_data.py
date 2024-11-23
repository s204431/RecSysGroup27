import torch
import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from Dataloader import NRMSDataLoader


from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from UserEncoder import UserEncoder
from torch import nn
import time

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
    def __init__(self, DATASET, type, K = 4): #Type is "train" or "validation"
        start = time.time()
        PATH = Path(__file__).parent.resolve().joinpath("./ebnerd_data")
    
        df_history   = pd.read_parquet(PATH.joinpath(DATASET, type, "history.parquet"))
        df_behaviors = pd.read_parquet(PATH.joinpath(DATASET, type, "behaviors.parquet"))
        df_articles  = pd.read_parquet(PATH.joinpath(DATASET, "articles.parquet"))
        print("Time to read files: ", time.time() - start)
        self.df_data = df_behaviors[['user_id', 'article_ids_inview', 'article_ids_clicked']].copy()

        article_ids_fixed_list = []
        for index, row in self.df_data.iterrows():
            user_id = row['user_id']
            #TODO: Next line is very slow!
            article_ids_fixed_for_user = df_history[df_history['user_id'] == user_id]['article_id_fixed'].tolist()
            article_ids_fixed_list.append(article_ids_fixed_for_user)
        
        print("Time: ", time.time() - start)
        self.df_data['article_ids_fixed'] = article_ids_fixed_list
        self.df_data['article_ids_fixed'] = self.df_data['article_ids_fixed'].apply(lambda x: np.concatenate(x).tolist() if isinstance(x, list) else x)

        for index, row in self.df_data.iterrows():
            article_ids_fixed = set(row['article_ids_fixed'])  
            article_ids_clicked = set(row['article_ids_clicked'])  
            remaining_articles = article_ids_fixed.difference(article_ids_clicked)
            self.df_data.at[index, 'article_ids_fixed'] = list(remaining_articles)

        print("Time after second for loop: ", time.time() - start)
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

        print("Time after third for loop: ", time.time() - start)
        self.df_data['gt_position'] = self.df_data.apply(find_position, axis=1)
        self.article_dict = pd.Series(df_articles['title'].values, index=df_articles['article_id']).to_dict()

        def replace_ids_with_titles(article_ids):
            return [self.article_dict.get(article_id, article_id) for article_id in article_ids]
        
        self.df_data['article_ids_inview'] = self.df_data['article_ids_inview'].apply(replace_ids_with_titles)
        self.df_data['article_ids_clicked'] = self.df_data['article_ids_clicked'].apply(replace_ids_with_titles)
        self.df_data['article_ids_fixed'] = self.df_data['article_ids_fixed'].apply(replace_ids_with_titles)
        print("Final time: ", time.time() - start)


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

#Parameters
dataset_name = 'ebnerd_demo'
k = 4
batch_size = 64
h = 16
dropout = 0.2

learning_rate = 1e-3
num_epochs = 10

validate_every = 50
validation_size = 1000

dataset = ArticlesDatasetTraining(dataset_name, 'train', K=k)
val_dataset = ArticlesDatasetTraining(dataset_name, 'validation', K=k)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=list)
user_encoder = UserEncoder(h=h, dropout=dropout)

optimizer = torch.optim.Adam(user_encoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
user_encoder.train()
train_losses = []
validation_losses = []
train_accuracies = []
validation_accuracies = []
for i in range(0, num_epochs):
    n_batches_finished = 0
    validation_index = 0
    accuracies = []
    losses = []
    for batch in train_loader:
        batch_outputs = []
        batch_targets = []
        for history, sample, target, gt_position in batch:
            if math.isnan(gt_position):
                continue
            history = getLastN(history, 10)
            #print(len(history), len(sample), len(target), gt_position)
            output = user_encoder(history=history, targets=sample)
            batch_outputs.append(output)
            batch_targets.append(torch.tensor(int(gt_position)))
        loss = criterion(torch.stack(batch_outputs), torch.stack(batch_targets))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = accuracy(batch_outputs, batch_targets)
        accuracies.append(acc)
        losses.append(loss.data.numpy())
        n_batches_finished += 1
        print("Number of batches finished: ", n_batches_finished)
        print("Batch loss: ", float(loss.data.numpy()))
        print("Batch accuracy: ", acc)
        print("Average accuracy so far in epoch: ", sum(accuracies)/len(accuracies))
        print()
        if n_batches_finished % validate_every == 0:
            break
    user_encoder.eval()
    print("Validation in epoch", i)
    batch = random.sample(range(0, len(val_dataset)), validation_size)
    batch_outputs = []
    batch_targets = []
    for sample in batch:
        history, sample, target, gt_position = val_dataset[sample]
        if math.isnan(gt_position):
            continue
        history = getLastN(history, 10)
        #print(len(history), len(sample), len(target), gt_position)
        output = user_encoder(history=history, targets=sample)
        batch_outputs.append(output)
        batch_targets.append(torch.tensor(int(gt_position)))
        validation_index += 1
    loss = criterion(torch.stack(batch_outputs), torch.stack(batch_targets))
    acc = accuracy(batch_outputs, batch_targets)
    print("Validation loss: ", loss.data.numpy())
    print("Validation accuracy: ", acc)
    print()
    train_losses.append(sum(losses)/len(losses))
    validation_losses.append(loss.data.numpy())
    train_accuracies.append(sum(accuracies)/len(accuracies))
    validation_accuracies.append(acc)
    user_encoder.train()

print("Validation accuracies: ", validation_accuracies)
#Plot results
iterations = [i for i in range(0, num_epochs)]
fig = plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(iterations, train_losses, label='train_loss')
plt.plot(iterations, validation_losses, label='valid_loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(iterations, train_accuracies, label='train_accs')
plt.plot(iterations, validation_accuracies, label='valid_accs')
plt.legend()
plt.show()

