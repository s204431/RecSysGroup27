import torch
import spacy
from NRMS import NRMS
from NRMSExtended import NRMSExtended
from Training import train, tuneParameters, testOnWholeDataset
from Testing import runOnTestSet

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Choose execution type
load_model = False
save_model = True
testing = False

# Model parameters
nlp = spacy.load("da_core_news_lg") # Load danish language model
dropout = 0.2
h = 16                              # Number of attention heads in multi-head self-attention

# Train parameters
dataset_name = 'ebnerd_small'   # Name of the dataset used for training and validating
k = 4                           # Number of negative samples to pair with each positive sample
batch_size = 64
weight_decay = 0.0
learning_rate = 1e-3            
history_size = 30               # Number of user history entries after truncation or padding
max_title_size = 20             # Number of tokens in an article title after truncation or padding
num_epochs = 100                # Not really used
validate_every = 200            # How many train batches between validations
validation_batch_size = 200     # The batch size for validation
n_validation_batches = 50       # How many batches to run for each validation
max_batches = 5000              # Use this if you want to end the training early

with_time_embeddings = True

def main():
    if with_time_embeddings:
        model = NRMSExtended(nlp, h=h, dropout=dropout).to(DEVICE)
    else:
        model = NRMS(nlp, h=h, dropout=dropout).to(DEVICE)

    if load_model:
        model.load_state_dict(torch.load('model_best_nrms_30_20.pth', map_location=DEVICE))

    #with torch.no_grad(): #Test on whole validation set
        #testOnWholeDataset(model, "ebnerd_small", "validation", history_size, max_title_size, nlp, batch_size=validation_batch_size, with_time_embeddings=with_time_embeddings)

    if not testing:
        train(
            model=model, 
            dataset_name=dataset_name, 
            k=k, 
            batch_size=batch_size, 
            weight_decay=weight_decay, 
            learning_rate=learning_rate, 
            history_size=history_size, 
            max_title_size=max_title_size, 
            nlp=nlp, 
            save_model=save_model, 
            num_epochs=num_epochs,
            validate_every=validate_every,
            validation_batch_size=validation_batch_size,
            n_validation_batches=n_validation_batches,
            max_batches=max_batches,
            with_time_embeddings=with_time_embeddings
            )

    else:
        with torch.no_grad():
            runOnTestSet(model, history_size, max_title_size, nlp, batch_size=validation_batch_size, with_time_embeddings=with_time_embeddings)


if __name__ == '__main__':
    main()