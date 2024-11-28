import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from UserEncoder import UserEncoder
from torch import nn
from sklearn.metrics import roc_auc_score
from Dataloading import ArticlesDatasetTraining
from Testing import runOnTestSet
import spacy
from torch.nn.utils.rnn import pad_sequence

nlp = spacy.load("da_core_news_md")  # Load danish model

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

def replace_titles_with_tokens(article_titles, nlp, max_vocab_size, batch_size=1000):
    with nlp.select_pipes(enable="tokenizer"):
        return [[min(max_vocab_size, token.rank) for token in doc] for doc in nlp.pipe(article_titles, batch_size=batch_size)]
    
def pad_tokens(tokens, token_length, padding_value):
    if len(tokens) > token_length:
        return tokens[:token_length]
    padding = [padding_value] * (token_length - len(tokens))
    return tokens + padding

def pad_token_list(token_list, token_length, padding_value, list_length):
    new_token_list = [pad_tokens(tokens=tokens, token_length=token_length, padding_value=padding_value) for tokens in token_list]
    if len(new_token_list) > list_length:
        return new_token_list[:list_length]
    title_padding = [padding_value] * (list_length - len(new_token_list))
    return new_token_list + title_padding

def pad_token_list_only_tokens(token_list, token_length, padding_value):
    return [pad_tokens(tokens=tokens, token_length=token_length, padding_value=padding_value) for tokens in token_list]

def make_batch(batch):
    max_title_size = 20
    vocab_size = nlp.vocab.vectors.shape[0]
    unknown_token_vector = [vocab_size for _ in range(max_title_size)]
    batch_history = []
    batch_targets = []
    batch_gtpositions = []
    for user_id, inview, clicked in batch:
        history, targets, gt_position = getData(user_id, inview, clicked, dataset, history_size, k)
        if history == None:
            continue
        #output = user_encoder(history=history, targets=targets)
        #batch_outputs.append(output)
        history_tokens_ids = []
        for title in history:
            doc = nlp(title)
            #history_tokens_ids.append(([(token.rank if token.rank < vocab_size else vocab_size) for token in doc] + unknown_token_vector)[:max_title_size])
            history_tokens_ids.append(torch.tensor([(token.rank if token.rank < vocab_size else vocab_size) for token in doc]))

        target_tokens_ids = []
        for title in targets:
            doc = nlp(title)
            #target_tokens_ids.append(([(token.rank if token.rank < vocab_size else vocab_size) for token in doc] + unknown_token_vector)[:max_title_size])
            target_tokens_ids.append(torch.tensor([(token.rank if token.rank < vocab_size else vocab_size) for token in doc]))

        #print(pad_sequence(history_tokens_ids, batch_first=True, padding_value=vocab_size).shape)
        batch_history.append(pad_sequence(history_tokens_ids, batch_first=True, padding_value=vocab_size))
        batch_targets.append(pad_sequence(target_tokens_ids, batch_first=True, padding_value=vocab_size))
        batch_gtpositions.append(torch.tensor(int(gt_position)))
    batch_history = torch.stack(batch_history)
    batch_targets = torch.stack(batch_targets)
    batch_gtpositions = torch.stack(batch_gtpositions)
    return batch_history, batch_targets, batch_gtpositions


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

dataset = ArticlesDatasetTraining(dataset_name, 'train')
val_dataset = ArticlesDatasetTraining(dataset_name, 'validation')
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
n_batches_finished = 0
for i in range(0, num_epochs):
    accuracies = []
    losses = []
    aucs = []
    for batch in train_loader:

        '''
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
        '''
        batch_history, batch_targets, batch_gtpositions = make_batch(batch)

        batch_outputs = []
        for history, targets in zip(batch_history, batch_targets):
            output = user_encoder(history=history, targets=targets)
            batch_outputs.append(output)
        batch_outputs = torch.stack(batch_outputs)
        batch_targets = batch_gtpositions

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
        print("Average accuracy since last validation: ", sum(accuracies)/len(accuracies))
        if len(aucs) > 0:
            print("Average auc score since last validation: ", sum(aucs)/len(aucs))
        print()
        if n_batches_finished % validate_every == 0:
            user_encoder.eval()
            print("Validation number", n_batches_finished//validate_every)
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
            accuracies = []
            losses = []
            aucs = []
            user_encoder.train()
        if n_batches_finished >= max_batches:
            break
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