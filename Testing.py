from Dataloading import ArticlesDatasetTest
from Utils import sampleIndices, sampleHistory, replace_titles_with_tokens, pad_token_list, findMaxInviewInBatchTesting, convertOutput
import torch
from torch.utils.data import DataLoader
import scipy.stats as ss

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

def convertLines():
    with open('predictions.txt', 'rb') as file:
        content = file.read()
    content = content.replace(b'\r\n', b'\n')
    with open("predictions_2.txt", "wb") as file:
        file.write(content)

def getData(user_id, inview, inview_times, impression_time, dataset, history_size, k):
    history = dataset.history_dict[user_id]
    history_times = dataset.time_dict[user_id]
    inview_times = impression_time - inview_times
    sampled_indices = sampleIndices(history, history_size)
    history = [history[i] for i in sampled_indices]
    history_times = [history_times[i] for i in sampled_indices]
    time_difference_history = impression_time - history_times
    time_padding_list_history = [99999.0] * (history_size - len(history_times))
    time_difference_history = time_difference_history.tolist() + time_padding_list_history

    targets_time = inview_times
    time_padding_list_inview = [99999.0] * (k+1 - len(inview_times))
    targets_time = targets_time.tolist() + time_padding_list_inview

    return history, inview, time_difference_history, targets_time

def make_batch(batch, dataset, nlp, k, history_size, max_title_size):
    vocab_size = nlp.vocab.vectors.shape[0]
    batch_history = []
    batch_targets = []
    batch_history_times = []
    batch_inview_times = []
    for _, user_id, inview, inview_times, impression_time in batch:
        history, targets, history_time, targets_time = getData(user_id, inview, inview_times, impression_time, dataset, history_size, k)
        #print(history, targets)

        #history = replace_titles_with_tokens(history, nlp, vocab_size, history_size)
        batch_history.append(pad_token_list(history, max_title_size, vocab_size+1, history_size))

        #targets = replace_titles_with_tokens(targets, nlp, vocab_size, k+1)
        batch_targets.append(pad_token_list(targets, max_title_size, vocab_size+1, k+1))

        batch_history_times.append(history_time)
        batch_inview_times.append(targets_time)

    batch_history = torch.tensor(batch_history).to(DEVICE)
    batch_targets = torch.tensor(batch_targets).to(DEVICE)
    batch_history_times = torch.tensor(batch_history_times).to(DEVICE)
    batch_inview_times = torch.tensor(batch_inview_times).to(DEVICE)
    return batch_history, batch_targets, batch_history_times, batch_inview_times

def runOnTestSet(model, history_size, max_title_size, nlp, batch_size, with_time_embeddings):
    log_every = 10000//batch_size

    model.eval()
    test_dataset = ArticlesDatasetTest('ebnerd_testset', nlp)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=list)
    outputs = []
    iteration = 0
    
    for batch in validation_loader:
        k_batch = findMaxInviewInBatchTesting(batch)
        batch_history, batch_targets, batch_history_times, batch_inview_times = make_batch(batch, test_dataset, nlp, k_batch, history_size, max_title_size)
        if with_time_embeddings:
            batch_outputs = model(history=batch_history, targets=batch_targets, history_times=batch_history_times.float(), inview_times=batch_inview_times.float())
        else:
            batch_outputs = model(history=batch_history, targets=batch_targets)
        batch_outputs = batch_outputs.cpu().numpy()
        batch_outputs = convertOutput(batch_outputs, batch)
        
        for i in range(0, len(batch)):
            impression_id, _, _, _, _ = batch[i]
            output = batch_outputs[i]
            ranking = ss.rankdata(-output, method="ordinal").astype(int).tolist()
            outputs.append([[impression_id], ranking])
        iteration += 1

        if iteration % log_every == 0:
                print("Finished: " + str(len(outputs)) + "/" + str(len(test_dataset)))
                
    saveToFile(outputs)