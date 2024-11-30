from Dataloading import ArticlesDatasetTest
import random
from Utils import getRandomN, replace_titles_with_tokens, pad_token_list, findMaxInviewInBatchTesting, convertOutput
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from UserEncoder import UserEncoder
import spacy

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def listToString(lst):
    string = "["
    for i in range(0, len(lst)):
        string += str(lst[i])
        if i < len(lst)-1:
            string += ","
    return string + "]"

def saveToFile(outputs):
    log_every = 100000
    print("Saving file...")
    iteration = 0
    file = open("predictions.txt", "w")
    for output in outputs:
        file.write(str(output[0][0]) + " " + listToString(output[1]) + "\n")
        iteration += 1
        if iteration % log_every == 0:
            print("Saved lines: ", iteration)
    file.close()

def getData(user_id, inview, dataset, history_size):
    history = dataset.history_dict[user_id]
    history = getRandomN(history, history_size)
    return history, inview

def make_batch(batch, dataset, nlp, k, history_size):
    max_title_size = 20
    vocab_size = nlp.vocab.vectors.shape[0]
    batch_history = []
    batch_targets = []
    for _, user_id, inview in batch:
        history, targets = getData(user_id, inview, dataset, history_size)
        #print(history, targets)

        #history = replace_titles_with_tokens(history, nlp, vocab_size, history_size)
        batch_history.append(pad_token_list(history, max_title_size, vocab_size, history_size))

        #targets = replace_titles_with_tokens(targets, nlp, vocab_size, k+1)
        batch_targets.append(pad_token_list(targets, max_title_size, vocab_size, k+1))

    batch_history = torch.tensor(batch_history).to(DEVICE)
    batch_targets = torch.tensor(batch_targets).to(DEVICE)
    return batch_history, batch_targets

def runOnTestSet(user_encoder, history_size, nlp):
    log_every = 50
    batch_size = 200

    user_encoder.eval()
    test_dataset = ArticlesDatasetTest('ebnerd_testset')
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=list)
    outputs = []
    iteration = 0
    
    for batch in validation_loader:
        k_batch = findMaxInviewInBatchTesting(batch)
        #start = time.time()
        batch_history, batch_targets = make_batch(batch, test_dataset, nlp, k_batch, history_size)
        #print("Time to make batch: ", time.time() - start)
        #start = time.time()
        batch_outputs = user_encoder(history=batch_history, targets=batch_targets)
        #batch_outputs = torch.tensor([[random.random() for _ in range(k_batch)] for _ in batch_history])
        #print("Time for batch: ", time.time() - start)
        #start = time.time()
        batch_outputs = batch_outputs.cpu().numpy()
        batch_outputs = convertOutput(batch_outputs, batch)
        
        for i in range(0, len(batch)):
            impression_id, _, inview = batch[i]
            output = batch_outputs[i]
            sorted_output = [x for _, x in sorted(zip(output, range(1, len(inview)+1)), reverse=True)]
            outputs.append([[impression_id], sorted_output])
        iteration += 1

        #print("Time to convert output: ", time.time() - start)

        if iteration % log_every == 0:
                print("Finished: ", len(outputs))
                #print("Test iteration", iteration)
                #break
    saveToFile(outputs)


nlp = spacy.load("da_core_news_md")  # Load danish model

h = 16
dropout = 0.2
history_size = 10
user_encoder = UserEncoder(h=h, dropout=dropout).to(DEVICE)
user_encoder.load_state_dict(torch.load('model.pth', map_location=DEVICE))
with torch.no_grad():
    runOnTestSet(user_encoder, history_size, nlp)