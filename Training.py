import torch
import random
from torch.utils.data import DataLoader
from NRMSExtended import NRMSExtended
from torch import nn
from Dataloading import ArticlesDatasetTraining
import spacy
from GitMetrics import AucScore
from Utils import sampleIndices, sampleHistory, replace_titles_with_tokens, pad_token_list, findMaxInviewInBatch, convertOutputAndgtPositions, convertgtPositionsToVec
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def accuracy(outputs, targets):
    nCorrect = 0
    for i in range(0, len(targets)):
        pred = torch.argmax(outputs[i])
        if pred == targets.cpu().numpy()[i]:
            nCorrect += 1
    return nCorrect/len(targets)

def getData(user_id, inview, inview_times, impression_time, clicked, clicked_times, dataset, history_size, k, negative_sampling=True):
    history = dataset.history_dict[user_id]
    history_times = dataset.time_dict[user_id]
    clicked = clicked[0]
    clicked_time = impression_time - clicked_times[0]
    inview_times = impression_time - inview_times
    if clicked in inview:
        inview.remove(clicked)
    if negative_sampling:
        if k > len(inview):
            return None, None, None
        indices = random.sample(range(len(inview)), k)
        targets = [inview[i] for i in indices]
        targets_time = [inview_times[i] for i in indices]
    else:
        targets = inview
        targets_time = inview_times
        time_padding_list_inview = [99999.0] * (k - len(inview_times))
        targets_time = targets_time.tolist() + time_padding_list_inview
    gt_position = random.randrange(0, len(targets)+1)
    targets.insert(gt_position, clicked)
    targets_time.insert(gt_position, clicked_time)
    sampled_indices = sampleIndices(history, history_size)
    history = [history[i] for i in sampled_indices]
    history_times = [history_times[i] for i in sampled_indices]
    time_difference_history = impression_time - history_times
    time_padding_list_history = [99999.0] * (history_size - len(history_times))
    time_difference_history = time_difference_history.tolist() + time_padding_list_history

    return history, targets, gt_position, time_difference_history, targets_time

def make_batch(batch, k, history_size, max_title_size, dataset, nlp, negative_sampling=True):
    vocab_size = nlp.vocab.vectors.shape[0]
    batch_history = []
    batch_targets = []
    batch_gtpositions = []
    batch_history_times = []
    batch_inview_times = []
    for user_id, inview, inview_times, impression_time, clicked, clicked_times in batch:
        history, targets, gt_position, history_time, targets_time = getData(user_id, inview, inview_times, impression_time, clicked, clicked_times, dataset, history_size, k, negative_sampling)
        if history != None:
            batch_history.append(pad_token_list(history, max_title_size, vocab_size+1, history_size))
            batch_targets.append(pad_token_list(targets, max_title_size, vocab_size+1, k+1))

            batch_gtpositions.append(int(gt_position))
            batch_history_times.append(history_time)
            batch_inview_times.append(targets_time)
    batch_history = torch.tensor(batch_history).to(DEVICE)
    batch_targets = torch.tensor(batch_targets).to(DEVICE)
    batch_gtpositions = torch.tensor(batch_gtpositions).to(DEVICE)
    batch_history_times = torch.tensor(batch_history_times).to(DEVICE)
    batch_inview_times = torch.tensor(batch_inview_times).to(DEVICE)
    return batch_history, batch_targets, batch_gtpositions, batch_history_times, batch_inview_times

def testOnWholeDataset(model, dataset_name, dataset_type, history_size, max_title_size, nlp, batch_size, with_time_embeddings):
    log_every = 10000//batch_size
    model.eval()
    test_dataset = ArticlesDatasetTraining(dataset_name, dataset_type, nlp)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=list)
    outputs = []
    targets = []
    iteration = 0
    for batch in dataloader:
        k_batch = findMaxInviewInBatch(batch)
        batch_history, batch_targets, batch_gtpositions, batch_history_times, batch_inview_times = make_batch(batch, k_batch, history_size, max_title_size, test_dataset, nlp, negative_sampling=False)
        if with_time_embeddings:
            batch_outputs = model(history=batch_history, targets=batch_targets, history_times=batch_history_times.float(), inview_times=batch_inview_times.float())
        else:
            batch_outputs = model(history=batch_history, targets=batch_targets)
        batch_targets = batch_gtpositions
        batch_outputs, batch_targets = convertOutputAndgtPositions(batch_outputs, batch_targets, batch)
        for i in range(0, len(batch)):
            outputs.append(batch_outputs[i])
            targets.append(batch_targets[i])
        iteration += 1
        if iteration % log_every == 0:
                print("Finished: " + str(len(outputs)) + "/" + str(len(test_dataset)))
    auc_metric = AucScore()
    auc_score = auc_metric.calculate(targets, outputs)
    print("Final AUC score on whole dataset: ", auc_score)
    return auc_score

def train(
        model, 
        dataset_name='ebnerd_large', 
        k=4, 
        batch_size=64, 
        weight_decay=0.0, 
        learning_rate=1e-3, 
        history_size=30, 
        max_title_size=20, 
        nlp=spacy.load("da_core_news_lg"), 
        save_model=False,
        num_epochs=100,
        validate_every=200,
        validation_batch_size=200,
        n_validation_batches=50,
        max_batches=50000,
        with_time_embeddings=True
        ):
    
    train_dataset = ArticlesDatasetTraining(dataset_name, 'train', nlp)
    val_dataset = ArticlesDatasetTraining(dataset_name, 'validation', nlp)
    #val_index_subset = random.sample(range(0, len(val_dataset)), validation_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=list)
    validation_loader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=True, collate_fn=list)
    #model = NRMS(h=h, dropout=dropout).to(DEVICE)
    #model.load_state_dict(torch.load('model.pth', map_location=DEVICE)) #Used to load the model from file

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()
    model.train()
    n_batches_finished = 0
    auc_metric = AucScore()
    train_aucs_overall = []
    train_losses_overall = []
    val_aucs_overall = []
    val_losses_overall = []
    iterations = []
    best_val_auc = 0.0
    for i in range(0, num_epochs):
        accuracies = []
        losses = []
        train_outputs = []
        train_gt_positions = []
        for batch in train_loader:

            batch_history, batch_targets, batch_gtpositions, batch_history_times, batch_inview_times = make_batch(batch, k, history_size, max_title_size, train_dataset, nlp)
            if with_time_embeddings:
                batch_outputs = model(history=batch_history, targets=batch_targets, history_times=batch_history_times.float(), inview_times=batch_inview_times.float())
            else:
                batch_outputs = model(history=batch_history, targets=batch_targets)

            loss = criterion(batch_outputs, batch_gtpositions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs_np = torch.softmax(batch_outputs, dim=1).cpu().detach().numpy()
            targets_np = convertgtPositionsToVec(batch_gtpositions, k+1)
            for output in outputs_np:
                train_outputs.append(output)
            for target in targets_np:
                train_gt_positions.append(target)
            acc = accuracy(batch_outputs, batch_gtpositions)
            accuracies.append(acc)
            losses.append(loss.cpu().detach().numpy()/batch_size)
            n_batches_finished += 1
            if n_batches_finished % validate_every == 0 or n_batches_finished == 1: #We also validate after the first batch to see amount of learning at the start
                model.eval()
                n_batches_finished_val = 0
                val_outputs = []
                val_gtpositions = []
                val_loss = 0
                print("Validation number " + str((n_batches_finished//validate_every) + 1) + "/" + str((max_batches//validate_every) + 1))
                print("Validation datapoints: ", (n_validation_batches*validation_batch_size))
                for batch in validation_loader:
                    batch_outputs = []
                    batch_targets = []

                    k_batch = findMaxInviewInBatch(batch)
                    batch_history, batch_targets, batch_gtpositions, batch_time_differences, batch_inview_times = make_batch(batch, k_batch, history_size, max_title_size, val_dataset, nlp, negative_sampling=False)
                    with torch.no_grad():
                        if with_time_embeddings:
                            batch_outputs = model(history=batch_history, targets=batch_targets, history_times=batch_time_differences.float(), inview_times=batch_inview_times.float())
                        else:
                            batch_outputs = model(history=batch_history, targets=batch_targets)
                        val_loss += criterion(batch_outputs, batch_gtpositions).cpu().numpy()
                    batch_outputs, batch_gtpositions = convertOutputAndgtPositions(batch_outputs, batch_gtpositions, batch)
                    for i in range(0, len(batch)):
                        val_outputs.append(batch_outputs[i])
                        val_gtpositions.append(batch_gtpositions[i])
                    n_batches_finished_val += 1
                    if (n_batches_finished_val >= n_validation_batches):
                        break
                
                val_aucscore = auc_metric.calculate(val_gtpositions, val_outputs)
                train_aucscore = auc_metric.calculate(train_gt_positions, train_outputs)
                print("Validation auc: ", val_aucscore)
                print("Validation loss: ", val_loss/len(val_outputs))
                print("Train auc: ", train_aucscore)
                print("Average train loss: ", sum(losses)/len(losses))
                print("Average train accuracy: ", sum(accuracies)/len(accuracies))
                print()
                train_aucs_overall.append(train_aucscore)
                train_losses_overall.append((sum(losses)/len(losses)).item())
                val_aucs_overall.append(val_aucscore)
                val_losses_overall.append((val_loss/len(val_outputs)).item())
                iterations.append(n_batches_finished)
                if val_aucscore > best_val_auc and save_model:
                    best_val_auc = val_aucscore
                    torch.save(model.state_dict(), 'model_best.pth')
                accuracies = []
                losses = []
                train_outputs = []
                train_gt_positions = []

                # Plot results
                plt.clf()   # Clears current figure
                plt.subplot(1, 2, 1)
                plt.plot(iterations, train_aucs_overall, label='train_AUC_scores')
                plt.plot(iterations, val_aucs_overall, label='valid_AUC_scores')
                plt.legend()
                plt.show()
                clear_output(wait=True)

                model.train()
            if n_batches_finished >= max_batches:
                break
        if n_batches_finished >= max_batches:
            break
    
    if save_model:
        torch.save(model.state_dict(), 'model.pth')
    
    #with torch.no_grad(): #Test on whole validation set
    #    final_auc = testOnWholeDataset(model, "ebnerd_small", "validation", history_size, max_title_size, nlp, batch_size=validation_batch_size, with_time_embeddings=with_time_embeddings)
    
    #model.load_state_dict(torch.load('model_best.pth', map_location=DEVICE))

    #with torch.no_grad(): #Test on whole validation set
    #    best_auc = testOnWholeDataset(model, "ebnerd_small", "validation", history_size, max_title_size, nlp, batch_size=validation_batch_size, with_time_embeddings=with_time_embeddings)
    
    #file = open("results.txt", "w")
    #file.write("Weight Decay " + str(weight_decay) + ", Learning rate " + str(learning_rate) + ", History size " + str(history_size) + ", Max title size " + str(max_title_size))
    #file.write("\nTrain aucs ")
    #file.write(str(train_aucs_overall))
    #file.write("\nTrain losses ")
    #file.write(str(train_losses_overall))
    #file.write("\nValidation aucs ")
    #file.write(str(val_aucs_overall))
    #file.write("\nValidation losses ")
    #file.write(str(val_losses_overall))
    #file.write("\nFinal AUC score on whole small validation set: ")
    #file.write(str(final_auc))
    #file.write("\nAUC score for best version: ")
    #file.write(str(best_auc))
    #file.close()

def tuneParameters(
        nlp, 
        dataset_name='ebnerd_large', 
        k=4,
        batch_size=64,
        num_epochs=100,
        validate_every=200,
        validation_batch_size=200,
        n_validation_batches=50,
        max_batches=50000
        ):
    
    """Tries different values of parameters and prints results"""
    print("Tuning...")
    weight_decays = [0.001, 0.00001]
    learning_rates = [1e-3]
    dropouts = [0.2]
    history_sizes = [20, 30]
    max_titles_sizes = [30]
    df = pd.DataFrame(columns=["weight_decay", "learning_rate", "dropout", "history_size", "max_titles_size", "AUC_Score"])
    for wd in weight_decays:
        for lr in learning_rates:
            for dout in dropouts:
                for hs in history_sizes:
                    for mts in max_titles_sizes:
                        model = NRMSExtended(nlp, 16, dout)
                        auc_score = train(model, dataset_name, k, batch_size, wd, lr, hs, mts, nlp, False, num_epochs, validate_every, validation_batch_size, n_validation_batches, max_batches)
                        new_row = {"weight_decay":wd, "learning_rate":lr, "dropout":dout, "history_size":hs, "max_titles_size":mts, "AUC_Score":auc_score}
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        df.to_excel("tuning2.xlsx")