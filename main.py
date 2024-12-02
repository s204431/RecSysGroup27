import torch
import torch.nn as nn
from NRMS import NRMS
import os
import spacy
from Training import train, tuneParameters
from Testing import runOnTestSet

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

load_model = True
save_model = False
testing = False

nlp = spacy.load("da_core_news_md")  # Load danish model
dropout = 0.2
weight_decay = 0.0
learning_rate = 1e-3
history_size = 10
max_title_size = 30
h = 16



def main():
    model = NRMS(nlp, h=h, dropout=dropout).to(DEVICE)

    if load_model:
        model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
    
    #tuneParameters(nlp) #Comment this out if you do not want to tune parameters

    if not testing:
        train(model, weight_decay, learning_rate, history_size, max_title_size, nlp)

        if save_model:
            torch.save(model.state_dict(), 'model.pth')
    else:
        with torch.no_grad():
            runOnTestSet(model, history_size, max_title_size, nlp)



























if __name__ == '__main__':
    main()