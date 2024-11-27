import torch
import random
import numpy as np
import sys
import wandb

from torch.utils.data import DataLoader
from UserEncoder import UserEncoder
from torch import nn
from sklearn.metrics import roc_auc_score
from Dataloading import ArticlesDatasetTraining
from Testing import runOnTestSet

def getLastN(lst, N):
    if len(lst) < N:
        return lst
    else:
        return lst[-N:]

def getRandomN(lst, N):
    if len(lst) < N:
        return lst
    else:
        return random.sample(lst, N)

def accuracy(outputs, targets):
    pred = torch.argmax(outputs, dim=1)  
    correct = torch.eq(pred, targets).sum().item() 
    return correct / targets.size(0)


def getData(user_id, inview, clicked, dataset, history_size, k=0):
    history = dataset.history_dict[user_id]
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
    if k > 0:
        gt_position = random.randrange(0, k+1)
    else:
        gt_position = random.randrange(0, len(inview)+1)
    targets.insert(gt_position, clicked)
    #history = getLastN(history, history_size)
    history = getRandomN(history, history_size)
    return history, targets, gt_position

#Parameters
dataset_name = 'ebnerd_small'
k = 4
batch_size = 64
h = 16
dropout = 0.2

learning_rate = 1e-3
num_epochs = 1

validate_every = 10000000
validation_size = 10
validation_number = 1
max_batches = 2000000 #100000000 #Use this if you want to end the training early

history_size = 10


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def training(user_encoder, train_dataset, train_loader, val_dataset, val_loader, optimizer, criterion, history_size, experiment_name):

    wandb.init(
        project = 'News prediction',
        name=experiment_name,
        config = {
            "num_epochs": num_epochs,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
    )

    train_loss = 0.0
    train_accuracies = 0.0
    train_aucs = 0.0
    n_batches_finished = 0

    for i in range(0, num_epochs):

        accuracies = []
        losses = []
        aucs = []
        user_encoder.train()

        for batch in train_loader:
            #print('training')

            batch_outputs = []
            batch_targets = []

            for user_id, inview, clicked in batch:
                history, targets, gt_position = getData(user_id, inview, clicked, train_dataset, history_size, k)
                if history == None:
                    continue
                output = user_encoder(history=history, targets=targets)
                batch_outputs.append(output)
                batch_targets.append(torch.tensor(int(gt_position)))

            batch_outputs = torch.stack(batch_outputs).to(DEVICE)
            batch_targets = torch.stack(batch_targets).to(DEVICE)
            loss = criterion(batch_outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_accuracies += accuracy(batch_outputs, batch_targets)
   
            if len(torch.unique(batch_targets)) == k + 1 == k+1:
                batch_targets_cpu = batch_targets.cpu().detach().numpy()
                batch_outputs_softmax = torch.softmax(batch_outputs, dim=1).cpu().detach().numpy()

                # Calculate AUC score and accumulate it
                train_aucs += roc_auc_score(batch_targets_cpu, batch_outputs_softmax, multi_class='ovr')
            

            n_batches_finished += 1

            wandb.log({
                'Finished batches': n_batches_finished,
                'Average train loss': train_loss/n_batches_finished,
                'Average train auc': train_aucs/n_batches_finished,
                'Average train accuracy': train_accuracies/n_batches_finished,
            })
            
      
            if n_batches_finished % validate_every == 0:
                
                val_count = 1
                print('validation')
                val_loss = 0
                val_accuracies = 0
                val_aucs = 0
                user_encoder.eval()

                with torch.no_grad():
                    
                    for user_ids, inview, clicked in val_loader:

                        batch_outputs = []
                        batch_targets = []
          
                        for user_id, inview, clicked in zip(user_ids, inview, clicked):
                            history, targets, gt_position = getData(user_id, inview, clicked, val_dataset, history_size, 0)
                            if history == None:
                                continue
                            output = user_encoder(history=history, targets=targets)
                            batch_outputs.append(output)
                            batch_targets.append(torch.tensor(int(gt_position)))

                        batch_outputs = torch.stack(batch_outputs).to(DEVICE)
                        batch_targets = torch.stack(batch_targets).to(DEVICE)
                        loss = criterion(batch_outputs, batch_targets)

                        val_loss += loss
                        val_accuracies += accuracy(batch_outputs, batch_targets)

                        if len(torch.unique(batch_targets)) == k + 1 == k+1:
                            batch_targets_cpu = batch_targets.cpu().detach().numpy()
                            batch_outputs_softmax = torch.softmax(batch_outputs, dim=1).cpu().detach().numpy()

                            # Calculate AUC score and accumulate it
                            val_aucs += roc_auc_score(batch_targets_cpu, batch_outputs_softmax, multi_class='ovr')

                        if val_count > validation_size:
                            break
                        print(val_count)
                        val_count += 1


                    wandb.log({
                        'Validation number': validation_number,
                        'Average val loss': val_loss/val_count,
                        'Average val auc': val_aucs/val_count,
                        'Average val accuracy': val_accuracies/val_count,
                    })


            if n_batches_finished >= max_batches:
                break
        if n_batches_finished >= max_batches:
            break

    ##Release train and validation datasets from memory before testing
    #dataset = None
    #val_dataset = None
    #train_loader = None
#
    ##Testing
    #with torch.no_grad():
    #    runOnTestSet(user_encoder, history_size)