import torch
import random
import numpy as np
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from UserEncoder import UserEncoder
from torch import nn
from sklearn.metrics import roc_auc_score
from Dataloading import ArticlesDatasetTraining, ArticlesDatasetTest

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
    nCorrect = 0
    for i in range(0, len(targets)):
        pred = torch.argmax(outputs[i])
        if pred == targets.data.numpy()[i]:
            nCorrect += 1
    return nCorrect/len(targets)

def replace_ids_with_titles(article_dict, article_ids):
    return [article_dict.get(article_id) for article_id in article_ids]

def getData(user_id, inview, clicked, dataset, history_size, k=0):
    inview = inview.tolist()
    clicked = clicked.tolist()
    history = dataset.history_dict[user_id].tolist()
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
    clicked = replace_ids_with_titles(dataset.article_dict, [clicked])[0]
    targets = replace_ids_with_titles(dataset.article_dict, targets)
    if k > 0:
        gt_position = random.randrange(0, k+1)
    else:
        gt_position = random.randrange(0, len(inview)+1)
    targets.insert(gt_position, clicked)
    #history = getLastN(history, history_size)
    history = getRandomN(history, history_size)
    history = replace_ids_with_titles(dataset.article_dict, history)
    return history, targets, gt_position

#Parameters
dataset_name = 'ebnerd_small'
k = 4
batch_size = 64
h = 16
dropout = 0.2

learning_rate = 1e-3
num_epochs = 100

validate_every = 50
validation_size = 2000

history_size = 10

dataset = ArticlesDatasetTraining(dataset_name, 'train')
val_dataset = ArticlesDatasetTraining(dataset_name, 'validation')
#test_dataset = ArticlesDatasetTest('ebnerd_testset')
val_index_subset = random.sample(range(0, len(val_dataset)), validation_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=list)
user_encoder = UserEncoder(h=h, dropout=dropout)

optimizer = torch.optim.Adam(user_encoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
user_encoder.train()
train_losses = []
validation_losses = []
train_accuracies = []
validation_accuracies = []
train_aucs = []
validation_aucs = []
for i in range(0, num_epochs):
    n_batches_finished = 0
    validation_index = 0
    accuracies = []
    losses = []
    aucs = []
    for batch in train_loader:
        batch_outputs = []
        batch_targets = []
        for user_id, inview, clicked in batch:
            history, targets, gt_position = getData(user_id, inview, clicked, dataset, history_size, k)
            if history == None:
                continue
            output = user_encoder(history=history, targets=targets)
            batch_outputs.append(output)
            batch_targets.append(torch.tensor(int(gt_position)))
        batch_outputs = torch.stack(batch_outputs)
        batch_targets = torch.stack(batch_targets)
        loss = criterion(batch_outputs, batch_targets)
        print("Backtracking")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = accuracy(batch_outputs, batch_targets)
        accuracies.append(acc)
        if len(np.unique(batch_targets.data.numpy())) == k+1:
            aucscore = roc_auc_score(batch_targets.data.numpy(), torch.softmax(batch_outputs, dim=1).data.numpy(), multi_class='ovr')
            aucs.append(aucscore)
        else:
            aucscore = None
        losses.append(loss.data.numpy())
        n_batches_finished += 1
        print("Number of batches finished: ", n_batches_finished)
        print("Batch loss: ", float(loss.data.numpy()))
        print("Batch accuracy: ", acc)
        print("Batch auc: ", aucscore)
        print("Average accuracy so far in epoch: ", sum(accuracies)/len(accuracies))
        if len(aucs) > 0:
            print("Average auc score so far in epoch: ", sum(aucs)/len(aucs))
        print()
        if n_batches_finished % validate_every == 0:
            break
    user_encoder.eval()
    print("Validation in epoch", i)
    batch_outputs = []
    batch_targets = []
    for sample in val_index_subset:
        user_id, inview, clicked = val_dataset[sample]
        history, targets, gt_position = getData(user_id, inview, clicked, val_dataset, history_size, k)
        if history == None:
            continue
        output = user_encoder(history=history, targets=targets)
        batch_outputs.append(output)
        batch_targets.append(torch.tensor(int(gt_position)))
    batch_outputs = torch.stack(batch_outputs)
    batch_targets = torch.stack(batch_targets)
    loss = criterion(batch_outputs, batch_targets)
    acc = accuracy(batch_outputs, batch_targets)
    if len(np.unique(batch_targets.data.numpy())) == k+1:
        aucscore = roc_auc_score(batch_targets.data.numpy(), torch.softmax(batch_outputs, dim=1).data.numpy(), multi_class='ovr')
    else:
        aucscore = 0.0
    print("Validation loss: ", loss.data.numpy())
    print("Validation accuracy: ", acc)
    print("Validation auc: ", aucscore)
    print()
    train_losses.append(sum(losses)/len(losses))
    validation_losses.append(loss.data.numpy())
    train_accuracies.append(sum(accuracies)/len(accuracies))
    validation_accuracies.append(acc)
    if len(aucs) > 0:
        train_aucs.append(sum(aucs)/len(aucs))
    validation_aucs.append(aucscore)
    user_encoder.train()

print("Validation accuracies: ", validation_accuracies)
print("Validation aucs: ", validation_aucs)
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

