import torch
import torch.nn as nn
from UserEncoder import UserEncoder
import os
import spacy
from Training import train, tuneParameters
from Testing import runOnTestSet

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

load_model = False
save_model = True
testing = False

nlp = spacy.load("da_core_news_md")  # Load danish model
dropout = 0.2
weight_decay = 0.0
learning_rate = 1e-3
history_size = 10
max_title_size = 30
h = 16


#user_encoder = UserEncoder(4, 0.2)

def main():
    user_encoder = UserEncoder(h=h, dropout=dropout).to(DEVICE)

    if load_model:
        user_encoder.load_state_dict(torch.load('model.pth', map_location=DEVICE))
    
    #tuneParameters(user_encoder) #Comment this out if you do not want to tune parameters

    if not testing:
        train(user_encoder, weight_decay, learning_rate, dropout, history_size, max_title_size, nlp)

        if save_model:
            torch.save(user_encoder.state_dict(), 'model.pth')
    else:
        with torch.no_grad():
            runOnTestSet(user_encoder, history_size, max_title_size, nlp)


    """for epoch in range(1, 3):  # Eksempel med epoch fra 1 til 2
         base_filename = f"user_encoder_{test_name}_{epoch}.pth"
        filename = base_filename
        counter = 1

        # Tjek, om filen eksisterer, og find et ledigt navn
        while os.path.exists(filename):
            filename = f"user_encoder_{test_name}_{epoch}_v{counter}.pth"
            counter += 1

        torch.save(user_encoder.state_dict(), filename)
        print(f"Model gemt som: {filename}")

        filename = f"Models/user_encoder_{test_name}_{epoch}.pth"
        torch.save(user_encoder.state_dict(), filename)
        print(f"Model gemt som: {filename}")"""



























if __name__ == '__main__':
    main()