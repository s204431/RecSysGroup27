import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from UserEncoder import UserEncoder
from torch import nn
from sklearn.metrics import roc_auc_score
from Dataloading import ArticlesDatasetTraining
import wandb
#from Testing import runOnTestSet
import spacy
from torch.nn.utils.rnn import pad_sequence
from GitMetrics import AucScore, AccuracyScore
from Utils import getRandomN, replace_titles_with_tokens, pad_token_list, findMaxInviewInBatch, convertOutputAndgtPositions, convertgtPositionsToVec



####################################################
# Har delt med 10                                  #
# Fryser vægten for embeddings                     #
# Ændre modelnavn                                  #
# Ændre name i run wandb                           #
####################################################

nlp = spacy.load("da_core_news_md")  # Load danish model

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Parameters
dataset_name = 'ebnerd_small'
k = 4
h = 16
dropout = 0.2

train_batch_size = 64
val_batch_size = 64

learning_rate = 1e-3
num_epochs = 100 #Not really used



history_size = 10



def accuracy(outputs, targets):
    nCorrect = 0
    for i in range(0, len(targets)):
        pred = torch.argmax(outputs[i])
        if pred == targets.cpu().numpy()[i]:
            nCorrect += 1
    return nCorrect/len(targets)

def getData(user_id, inview, clicked, dataset, history_size, k, negative_sampling=True):
    history = dataset.history_dict[user_id]
    clicked = clicked[0]
    if clicked in inview:
        inview.remove(clicked)
    if negative_sampling:
        if k > len(inview):
            return None, None, None
        targets = random.sample(inview, k)
    else:
        targets = inview
    gt_position = random.randrange(0, len(targets)+1)
    targets.insert(gt_position, clicked)
    history = getRandomN(history, history_size)
    return history, targets, gt_position

def make_batch(batch, k, dataset, negative_sampling=True):
    max_title_size = 20
    vocab_size = nlp.vocab.vectors.shape[0]
    batch_history = []
    batch_targets = []
    batch_gtpositions = []
    for user_id, inview, clicked in batch:
        history, targets, gt_position = getData(user_id, inview, clicked, dataset, history_size, k, negative_sampling)
        if history != None:

            history = replace_titles_with_tokens(history, nlp, vocab_size, history_size)
            batch_history.append(pad_token_list(history, max_title_size, vocab_size, history_size))

            targets = replace_titles_with_tokens(targets, nlp, vocab_size, k+1)
            batch_targets.append(pad_token_list(targets, max_title_size, vocab_size, k+1))

            batch_gtpositions.append(int(gt_position))
    batch_history = torch.tensor(batch_history).to(DEVICE)
    batch_targets = torch.tensor(batch_targets).to(DEVICE)
    batch_gtpositions = torch.tensor(batch_gtpositions).to(DEVICE)
    return batch_history, batch_targets, batch_gtpositions


if __name__ == '__main__':

    torch.manual_seed(42)
    train_dataset = ArticlesDatasetTraining(dataset_name, 'train')
    val_dataset = ArticlesDatasetTraining(dataset_name, 'validation')

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=list, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, collate_fn=list, num_workers=4)
    user_encoder = UserEncoder(h=h, dropout=dropout).to(DEVICE)

    optimizer = torch.optim.Adam(user_encoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
   
    n_batches_finished = 0
    auc_metric = AucScore()
    acc_metric =  AccuracyScore()

    wandb.init(
        project="News_prediction",  # Set your W&B project name
        name='mads_run_frozen',     # Name of the experiment
        config={                    # Log hyperparameters
            "num_epochs": num_epochs,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
    )    

    for i in range(0, num_epochs):

        train_accuracies = 0.0
        train_losses = 0.0
        train_aucscores = 0.0
        train_count = 0

        user_encoder.train()
        

        for batch in train_loader:

            train_outputs = []
            train_gt_positions = []            

            batch_history, batch_targets, batch_gtpositions = make_batch(batch, k, train_dataset)
            batch_outputs = user_encoder(history=batch_history, targets=batch_targets)
            batch_targets = batch_gtpositions

            loss = criterion(batch_outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs_np = batch_outputs.cpu().detach().numpy()
            targets_np = convertgtPositionsToVec(batch_targets, k+1)
            for output in outputs_np:
                train_outputs.append(output)
            for target in targets_np:
                train_gt_positions.append(target)
            
            train_losses += loss.cpu().detach().numpy()

            train_accuracies += acc_metric.calculate(train_gt_positions, train_outputs)
            train_aucscores += auc_metric.calculate(train_gt_positions, train_outputs)
            train_count += 1

    
        wandb.log({
            "epoch": i,
            "train_count": train_count,
            "Train auc: ": train_aucscores/train_count,
            "Average train loss: ": train_losses/train_count,
            "Average train accuracy: ": train_accuracies/train_count
            })


        val_accuracies = 0.0
        val_losses = 0.0
        val_aucscores = 0.0
        val_count = 0

        user_encoder.eval()

        for batch in val_loader:

            batch_outputs = []
            batch_targets = []

            k_batch = findMaxInviewInBatch(batch)
            batch_history, batch_targets, batch_gtpositions = make_batch(batch, k_batch, val_dataset, negative_sampling=False)
            with torch.no_grad():
                batch_outputs = user_encoder(history=batch_history, targets=batch_targets)
                loss = criterion(batch_outputs, batch_gtpositions)

            val_losses += loss.cpu().detach().numpy()
            
            batch_targets = batch_gtpositions
            batch_outputs, batch_targets = convertOutputAndgtPositions(batch_outputs, batch_targets, batch)

            val_accuracies += acc_metric.calculate(batch_targets, batch_outputs)
            val_aucscores += auc_metric.calculate(batch_targets, batch_outputs)
            val_count += 1


        wandb.log({
            "epoch": i,
            "val_count": val_count,
            "val auc: ": val_aucscores/val_count,
            "Average val loss: ": val_losses/val_count,
            "Average val accuracy: ": val_accuracies/val_count
        })
          
        filename = f'Models/model_frozen{i}.pth'
        torch.save(user_encoder.state_dict(), filename)

    



    #Testing
    #with torch.no_grad():
        #runOnTestSet(user_encoder, history_size, nlp)