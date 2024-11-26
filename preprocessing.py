import spacy
import time
import torch
import pandas as pd
import random
import os
import pickle as pk

from pathlib import Path
from pathlib import Path
from multiprocessing import Pool

# Indlæs spaCy modellen
nlp = spacy.load("da_core_news_lg")  # Brug dansk model

# Definer DEVICE for GPU, hvis du bruger CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Funktion til at hente artikel embedding (uden klasse)
def get_article_embedding(article_id):

    title = article_dict.get(article_id, None)
    
    if not isinstance(title, str):
        title = str(title)  
    doc = nlp(title)  

    vectors = torch.tensor([token.vector/10 for token in doc])
    return vectors.unsqueeze(0)



# Funktion til at erstatte artikel-IDs med titler
def replace_ids_with_titles(article_dict, article_ids):
    """Erstat artikel-IDs med titler baseret på article_dict"""
    return [article_dict.get(article_id, "Unknown") for article_id in article_ids]

# Funktion til at få tilfældige N elementer fra en liste
def getRandomN(lst, N):
    if not isinstance(lst, list):  # Hvis lst ikke er en liste, gør det til en
        lst = list(lst)
    if len(lst) < N:
        return lst
    else:
        return random.sample(lst, N)

# Funktion til at hente data for en given bruger
def getData(user_id, inview, clicked, article_dict, history_dict, history_size, k=0):
    inview = inview.tolist()
    clicked = clicked.tolist()
    
    if user_id in history_dict:
        history = history_dict[user_id].tolist()
    else:
        history = []
    
    for id in clicked:
        if id in inview:
            inview.remove(id)
    clicked = clicked[0]  

    if k > 0:
        if k > len(inview):
            return None, None, None
        targets = random.sample(inview, k)
    else:
        targets = inview

    clicked_title = replace_ids_with_titles(article_dict, [clicked])[0]
    targets_titles = replace_ids_with_titles(article_dict, targets)

    if k > 0:
        gt_position = random.randrange(0, k+1)
    else:
        gt_position = random.randrange(0, len(inview)+1)

    targets_titles.insert(gt_position, clicked_title)

    history = getRandomN(history, history_size)  # Sørg for at history er en liste
    history_titles = replace_ids_with_titles(article_dict, history)

    return history_titles, targets_titles, torch.tensor(gt_position)





if __name__ == '__main__':
    start = time.time()
    DATASET = 'ebnerd_demo'
    PATH = Path(__file__).parent.resolve().joinpath("./ebnerd_data")

    df_history_train = pd.read_parquet(PATH.joinpath(DATASET, 'train', "history.parquet"))
    df_behaviors_train = pd.read_parquet(PATH.joinpath(DATASET, 'train', "behaviors.parquet"))
    df_articles_train = pd.read_parquet(PATH.joinpath(DATASET, "articles.parquet"))
    
    df_history_train = df_history_train[['user_id', 'article_id_fixed']]
    df_articles_train = df_articles_train[['article_id', 'title']]
    df_behaviors_train = df_behaviors_train[['user_id', 'article_ids_inview', 'article_ids_clicked']]

    article_dict = pd.Series(df_articles_train['title'].values, index=df_articles_train['article_id']).to_dict()
    history_dict = pd.Series(df_history_train['article_id_fixed'].values, index=df_history_train['user_id']).to_dict()


    new_data = []

    for idx, row in df_behaviors_train.iterrows():
        user_id = row['user_id']
        inview = row['article_ids_inview']
        clicked = row['article_ids_clicked']

        history_titles, targets_titles, gt_position = getData(user_id, inview, clicked, article_dict, history_dict, history_size=30, k=4)

        #print('History titles\n', history_titles)
        #print()
        #print('target titles\n', targets_titles)
        #print()
        #print('gt_position\n', gt_position)
        #print()

        if history_titles is not None and targets_titles is not None:
            history_embedding = [get_article_embedding(article_id) for article_id in history_titles]
            targets_embedding = [get_article_embedding(article_id) for article_id in targets_titles] 

        #print('History embedding\n', type(history_embedding[0]))
        #print()
        #print('target embedding\n', type(targets_embedding[0]))
        #print()
        #print('gt_position\n', type(gt_position))


        # Gem resultaterne i en liste, som senere konverteres til en DataFrame
            new_data.append({
                'user_id': user_id,
                'history_titles': history_titles,
                'history_embeddings': history_embedding,
                'target_titles': targets_titles,
                'target_embeddings': targets_embedding,
                'gt_position': gt_position,
            })
            print(f"Processed {idx} articles / {len(df_behaviors_train)}")

            if idx > 20:
                break

    print(f'Time for creating the dataset', time.time() - start)
    # Opret en DataFrame fra den nye liste
    df_new = pd.DataFrame(new_data)

    # Udskriv eller gem DataFrame'en som ønsket
    print(df_new.head())

 

    save_targets_path = Path(__file__).parent.resolve().joinpath("./ebnerd_prep")
    os.makedirs(save_targets_path, exist_ok=True)


    with open(os.path.join(save_targets_path, f'test_target_{DATASET}.pkl'), 'wb') as f:
        pk.dump(df_new, f)

    print(f"Fil gemt som {save_targets_path}/test_target_{DATASET}.pkl")
