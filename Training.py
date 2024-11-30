import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from UserEncoder import UserEncoder
from torch import nn
from sklearn.metrics import roc_auc_score
from Dataloading import ArticlesDatasetTraining
#from Testing import runOnTestSet
import spacy
from torch.nn.utils.rnn import pad_sequence
from GitMetrics import AucScore, AccuracyScore
from Utils import getRandomN, replace_titles_with_tokens, pad_token_list, findMaxInviewInBatch, convertOutputAndgtPositions, convertgtPositionsToVec
import time

nlp = spacy.load("da_core_news_md")  # Load danish model

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Parameters
dataset_name = 'ebnerd_small'
k = 4
batch_size = 64
h = 16
dropout = 0.2

learning_rate = 1e-3
num_epochs = 100 #Not really used

validate_every = 200 #How many train batches between validations
validation_batch_size = 200 #The batch size for validation
n_validation_batches = 50 #How many batches to run for each validation

max_batches = 5000 #Use this if you want to end the training early

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

            #history = replace_titles_with_tokens(history, nlp, vocab_size, history_size)
            batch_history.append(pad_token_list(history, max_title_size, vocab_size, history_size))

            #targets = replace_titles_with_tokens(targets, nlp, vocab_size, k+1)
            batch_targets.append(pad_token_list(targets, max_title_size, vocab_size, k+1))

            batch_gtpositions.append(int(gt_position))
    batch_history = torch.tensor(batch_history).to(DEVICE)
    batch_targets = torch.tensor(batch_targets).to(DEVICE)
    batch_gtpositions = torch.tensor(batch_gtpositions).to(DEVICE)
    return batch_history, batch_targets, batch_gtpositions

def testOnWholeDataset(user_encoder, dataset_name, dataset_type):
    log_every = 50
    batch_size = 200
    user_encoder.eval()
    test_dataset = ArticlesDatasetTraining(dataset_name, dataset_type)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=list)
    outputs = []
    targets = []
    iteration = 0
    for batch in dataloader:
        k_batch = findMaxInviewInBatch(batch)
        batch_history, batch_targets, batch_gtpositions = make_batch(batch, k_batch, test_dataset, negative_sampling=False)
        batch_outputs = user_encoder(history=batch_history, targets=batch_targets)
        batch_targets = batch_gtpositions
        batch_outputs, batch_targets = convertOutputAndgtPositions(batch_outputs, batch_targets, batch)
        for i in range(0, len(batch)):
            outputs.append(batch_outputs[i])
            targets.append(batch_targets[i])
        iteration += 1
        if iteration % log_every == 0:
                print("Finished: ", len(outputs))
    auc_metric = AucScore()
    auc_score = auc_metric.calculate(targets, outputs)
    print("Final AUC score on whole dataset: ", auc_score)

if __name__ == '__main__':

    train_dataset = ArticlesDatasetTraining(dataset_name, 'train')
    val_dataset = ArticlesDatasetTraining(dataset_name, 'validation')
    #val_index_subset = random.sample(range(0, len(val_dataset)), validation_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=list)
    validation_loader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=True, collate_fn=list)
    user_encoder = UserEncoder(h=h, dropout=dropout).to(DEVICE)
    #user_encoder.load_state_dict(torch.load('model.pth', map_location=DEVICE)) #Used to load the model from file

    optimizer = torch.optim.Adam(user_encoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    user_encoder.train()
    n_batches_finished = 0
    auc_metric = AucScore()
    for i in range(0, num_epochs):
        accuracies = []
        losses = []
        train_outputs = []
        train_gt_positions = []
        for batch in train_loader:

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
            acc = accuracy(batch_outputs, batch_targets)
            accuracies.append(acc)
            losses.append(loss.cpu().detach().numpy())
            n_batches_finished += 1
            if n_batches_finished % validate_every == 0 or n_batches_finished == 1: #We also validate after the first batch to see amount of learning at the start
                user_encoder.eval()
                n_batches_finished_val = 0
                val_outputs = []
                val_targets = []
                print("Validation number", n_batches_finished//validate_every)
                print("Validation datapoints: ", (n_validation_batches*validation_batch_size))
                for batch in validation_loader:
                    batch_outputs = []
                    batch_targets = []

                    k_batch = findMaxInviewInBatch(batch)
                    batch_history, batch_targets, batch_gtpositions = make_batch(batch, k_batch, val_dataset, negative_sampling=False)
                    with torch.no_grad():
                        batch_outputs = user_encoder(history=batch_history, targets=batch_targets)
                    batch_targets = batch_gtpositions
                    batch_outputs, batch_targets = convertOutputAndgtPositions(batch_outputs, batch_targets, batch)
                    for i in range(0, len(batch)):
                        val_outputs.append(batch_outputs[i])
                        val_targets.append(batch_targets[i])
                    n_batches_finished_val += 1
                    if (n_batches_finished_val >= n_validation_batches):
                        break

                val_aucscore = auc_metric.calculate(val_targets, val_outputs)
                train_aucscore = auc_metric.calculate(train_gt_positions, train_outputs)
                print("Validation auc: ", val_aucscore)
                print("Train auc: ", train_aucscore)
                print("Average train loss: ", sum(losses)/len(losses))
                print("Average train accuracy: ", sum(accuracies)/len(accuracies))
                print()
                accuracies = []
                losses = []
                train_outputs = []
                train_gt_positions = []
                user_encoder.train()
            if n_batches_finished >= max_batches:
                break
        if n_batches_finished >= max_batches:
            break

    torch.save(user_encoder.state_dict(), 'model.pth')
    
    with torch.no_grad(): #Test on whole validation set
        testOnWholeDataset(user_encoder, "ebnerd_small", "validation")