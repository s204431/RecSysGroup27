import torch
import time



def calculate_auc(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """
    Beregner gennemsnitlig AUC for labels og forudsigelser som 2D-tensors.
    Hver række i tensoren er et separat element, og padding (-1) ignoreres.
    """
    def _binary_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
        # Ignorer padding værdier (som er -1)
        mask = y_true != -1
        y_true = y_true[mask]
        y_score = y_score[mask]

        # Hvis der er ingen gyldige værdier (dvs. alle er -1), returner 0
        if len(y_true) == 0:
            return 0.0

        # Sorter efter score
        sorted_indices = torch.argsort(y_score, descending=True)
        y_true = y_true[sorted_indices]
        y_score = y_score[sorted_indices]

        # Beregn TPR (True Positive Rate) og FPR (False Positive Rate)
        positives = y_true.sum()
        negatives = len(y_true) - positives

        tpr = torch.cumsum(y_true, dim=0) / positives
        fpr = torch.cumsum(1 - y_true, dim=0) / negatives

        # AUC som areal under kurve (trapez)
        auc = torch.trapz(tpr, fpr)
        return auc.item()

    auc_scores = [
        _binary_auc(label, prediction)
        for label, prediction in zip(labels, predictions)
    ]
    return float(torch.tensor(auc_scores).mean())


def calculate_accuracy(labels: torch.Tensor, predictions: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Beregner gennemsnitlig accuracy for labels og forudsigelser som 2D-tensors.
    Hver række i tensoren er et separat element, og padding (-1) ignoreres.
    """
    def _binary_accuracy(y_true: torch.Tensor, y_score: torch.Tensor, threshold: float) -> float:
        # Ignorer padding værdier (som er -1)
        mask = y_true != -1
        y_true = y_true[mask]
        y_score = y_score[mask]

        # Hvis der er ingen gyldige værdier (dvs. alle er -1), returner 0
        if len(y_true) == 0:
            return 0.0

        y_pred = (y_score >= threshold).long()
        correct = (y_pred == y_true).sum().item()
        total = y_true.numel()
        return correct / total

    accuracies = [
        _binary_accuracy(label, prediction, threshold)
        for label, prediction in zip(labels, predictions)
    ]
    return float(torch.tensor(accuracies).mean())

