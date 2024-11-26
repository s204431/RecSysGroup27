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

validate_every = 50
validation_size = 500
max_batches = 100000000 #Use this if you want to end the training early

history_size = 10

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def training(user_encoder, train_dataset, train_loader, val_dataset, optimizer, criterion, history_size, experiment_name):

    wandb.init(
        project = 'News prediction',
        name=experiment_name,
        config = {
            "num_epochs": num_epochs,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
    )

    user_encoder.train()

    train_loss = 0.0
    train_accuracies = 0.0
    train_aucs = 0.0
    n_batches_finished = 0

    for i in range(0, num_epochs):

        accuracies = []
        losses = []
        aucs = []

        for batch in train_loader:

            batch_outputs = []
            batch_targets = []
            ja = 0
            for user_id, inview, clicked in batch:
                ja += 1
                print(ja)
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
                'Number of epochs': i+1
            })
            
            continue
            #if n_batches_finished % validate_every == 0:
            #    user_encoder.eval()
            #    print("Validation number", n_batches_finished//validate_every)
            #    batch_outputs = []
            #    batch_targets = []
            #    for sample in val_index_subset:
            #        user_id, inview, clicked = val_dataset[sample]
            #        history, targets, gt_position = getData(user_id, inview, clicked, val_dataset, history_size, k)
            #        if history == None:
            #            continue
            #        output = user_encoder(history=history, targets=targets)
            #        batch_outputs.append(output)
            #        batch_targets.append(torch.tensor(int(gt_position)))
            #    batch_outputs = torch.stack(batch_outputs)
            #    batch_targets = torch.stack(batch_targets)
            #    loss = criterion(batch_outputs, batch_targets)
            #    acc = accuracy(batch_outputs, batch_targets)
            #    if len(np.unique(batch_targets.data.numpy())) == k+1:
            #        aucscore = roc_auc_score(batch_targets.data.numpy(), torch.softmax(batch_outputs, dim=1).data.numpy(), multi_class='ovr')
            #    else:
            #        aucscore = 0.0
            #    train_losses += loss
            #    validation_losses.append(loss.data.numpy())
            #    train_accuracies.append(sum(accuracies)/len(accuracies))
            #    validation_accuracies.append(acc)
            #    if len(aucs) > 0:
            #        train_aucs.append(sum(aucs)/len(aucs))
            #    validation_aucs.append(aucscore)
            #    accuracies = []
            #    losses = []
            #    aucs = []
            #    user_encoder.train()
            #if n_batches_finished >= max_batches:
            #    break
        if n_batches_finished >= max_batches:
            break

    print("Validation accuracies: ", validation_accuracies)
    print("Validation aucs: ", validation_aucs)

    #Release train and validation datasets from memory before testing
    dataset = None
    val_dataset = None
    train_loader = None

    #Testing
    with torch.no_grad():
        runOnTestSet(user_encoder, history_size)