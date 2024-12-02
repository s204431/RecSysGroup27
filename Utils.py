import random
import numpy as np

def getLastN(lst, N):
    if len(lst) < N:
        return lst
    else:
        return lst[-N:]
    
def getFirstN(lst, N):
    if len(lst) < N:
        return lst
    else:
        return lst[:N]

def getRandomN(lst, N):
    if len(lst) < N:
        return lst
    else:
        return random.sample(lst, N)

def sampleHistory(history, history_times, N):
    return getRandomN(history, N)
    #return getLastN(history, N)
    #return getFirstN(history, N)

def sampleIndices(lst, N):
    return getRandomN(range(len(lst)), N)

def replace_titles_with_tokens(article_titles, nlp, max_vocab_size, batch_size=64):
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
    title_padding = [[padding_value for _ in range(token_length)] for _ in range(list_length - len(new_token_list))]
    return new_token_list + title_padding

def pad_token_list_only_tokens(token_list, token_length, padding_value):
    return [pad_tokens(tokens=tokens, token_length=token_length, padding_value=padding_value) for tokens in token_list]

def findMaxInviewInBatch(batch):
    maximum = 0
    for _, inview, _, _ in batch:
        if len(inview) > maximum:
            maximum = len(inview)
    return maximum

def findMaxInviewInBatchTesting(batch):
    maximum = 0
    for _, _, inview, _ in batch:
        if len(inview) > maximum:
            maximum = len(inview)
    return maximum

def convertgtPositionsToVec(batch_gt_positions, length):
    targets = []
    for gt_position in batch_gt_positions:
        gt_vec = np.zeros(length)
        gt_vec[gt_position] = 1
        targets.append(gt_vec)
    return targets

def convertOutput(batch_output, batch):
    outputs = []
    for i in range(0, len(batch_output)):
        output = batch_output[i]
        _, _, inview, _ = batch[i]
        output = output[:len(inview)]
        outputs.append(output)
        #outputs.append(np.exp(output)/sum(np.exp(output))) #Softmax
    return outputs

def convertOutputAndgtPositions(batch_output, batch_gt_positions, batch):
    outputs = []
    targets = []
    for i in range(0, batch_output.shape[0]):
        output = batch_output[i]
        _, inview, _, _ = batch[i]
        gt_position = batch_gt_positions[i]
        output = output[:len(inview)].cpu().numpy()
        outputs.append(np.exp(output)/sum(np.exp(output))) #Softmax
        gt_vec = np.zeros(len(inview))
        gt_vec[gt_position] = 1
        targets.append(gt_vec)
    return outputs, targets