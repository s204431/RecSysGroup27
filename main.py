import torch
import torch.nn as nn
import argparse
import random

from UserEncoder import UserEncoder
from Dataloading import ArticlesDatasetTraining, ArticlesDatasetValTest
from torch.utils.data import DataLoader
from Training import training

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    user_encoder = UserEncoder(args.h, args.dropout).to(DEVICE)

    train_dataset = ArticlesDatasetTraining(args.dataset)
    val_dataset = ArticlesDatasetValTest(args.dataset, 'validation')

    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4,
        collate_fn = list
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 1,
        collate_fn = list
    )

    optimizer = torch.optim.Adam(user_encoder.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()

    training(user_encoder, train_dataset, train_loader, val_dataset, optimizer, criterion, args.history_size, args.experiment_name)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train and evaluate a userencoder model.")

    # Existing arguments
    parser.add_argument('--experiment_name', type=str, default='experiment1', help='Name of the experiment')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Se seed to get same results')
    parser.add_argument('--h', type=int, default=16, help='Number of header')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout value')
    parser.add_argument('--dataset', type=str, default='ebnerd_small', help='Dropout value')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of the batch')
    parser.add_argument('--history_size', type=int, default=30, help='Number of articles used in userencoder from history')


    args = parser.parse_args()

    main(args)