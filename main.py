import torch
import torch.nn as nn
from NRMS import NRMS
from NRMSExtended import NRMSExtended
import os
import spacy
from Training import train, tuneParameters, testOnWholeDataset
from Testing import runOnTestSet

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

load_model = False
save_model = True
testing = False

nlp = spacy.load("da_core_news_lg")  # Load danish model
dropout = 0.2
weight_decay = 0.0
learning_rate = 1e-3
history_size = 30
max_title_size = 20
h = 16

def main():
    model = NRMSExtended(nlp, h=h, dropout=dropout).to(DEVICE)

    if load_model:
        model.load_state_dict(torch.load('model_best_nrms_30_20.pth', map_location=DEVICE))
    
    #tuneParameters(nlp) #Comment this out if you do not want to tune parameters

    #with torch.no_grad(): #Test on whole validation set
        #testOnWholeDataset(model, "ebnerd_small", "validation", history_size, max_title_size, nlp)

    if not testing:
        train(model, weight_decay, learning_rate, history_size, max_title_size, nlp, save_model=save_model)

    else:
        with torch.no_grad():
            runOnTestSet(model, history_size, max_title_size, nlp)



























if __name__ == '__main__':
    main()