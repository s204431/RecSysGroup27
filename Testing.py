from Dataloading import ArticlesDatasetTest
import random
import torch
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def getRandomN(lst, N):
    if len(lst) < N:
        return lst
    else:
        return random.sample(lst, N)

def listToString(lst):
    string = "["
    for i in range(0, len(lst)):
        string += str(lst[i])
        if i < len(lst)-1:
            string += ","
    return string + "]"

def saveToFile(outputs):
    file = open("predictions.txt", "w")
    for output in outputs:
        file.write(str(output[0][0]) + " " + listToString(output[1]) + "\n")
    file.close()

def runOnTestSet(user_encoder, history_size):
    print('start')
    log_every = 10000

    user_encoder.to(DEVICE)
    user_encoder.eval()
    test_dataset = ArticlesDatasetTest('ebnerd_testset')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=list, num_workers = 2)
    outputs = []
    iteration = 0
    for batch in test_loader:
        for impression_id, user_id, inview_titles, inview_ids in batch:
            history = test_dataset.history_dict[user_id]
            history = getRandomN(history, history_size)
            output = user_encoder(history=history, targets=inview_titles)
            #output = [random.random() for _ in range(0, len(inview_ids))] #This gives random outputs
            sorted_output = [x for _, x in sorted(zip(output, inview_ids.tolist()), reverse=True)]
            outputs.append([[impression_id], sorted_output])
            iteration += 1
            #if iteration % log_every == 0:
            print("Test iteration", iteration)
            #break
    saveToFile(outputs)
        