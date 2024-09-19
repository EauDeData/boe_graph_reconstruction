from tqdm import tqdm
import torch
import torch.nn as nn
import torch
from torch import Tensor
from typing import *
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score

def cosine_similarity_matrix(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    '''
    When using cosine similarity the constant value must be positive
    '''
    #Cosine sim:
    xn1, xn2 = torch.norm(x1, dim=dim), torch.norm(x2, dim=dim)
    x1 = x1 / torch.clamp(xn1, min=eps).unsqueeze(dim)
    x2 = x2 / torch.clamp(xn2, min=eps).unsqueeze(dim)
    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(1)

    sim = torch.tensordot(x1, x2, dims=([2], [2])).squeeze()

    sim = (sim + 1)/2 #range: [-1, 1] -> [0, 2] -> [0, 1]

    return sim

def euclidean_similarity_matrix(x1, x2, eps):
    return 1/(1+torch.cdist(x1, x2)+eps)

def euclidean_distance_matrix(x1, x2):
    return torch.cdist(x1, x2)

def sigmoid(x, k=1.0):
    exponent = -x/k
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1./(1. + torch.exp(exponent))
    return y

def batched_top_k(distance_matrix, gt, k = 1):
    sorted, _ = torch.sort(distance_matrix, dim = 1, descending = True)
    return sum(sorted[:, k - 1] <= distance_matrix[gt]) / sorted.shape[0]


def batched_matrix(x, y, dist=cosine_similarity_matrix, img2text=False):
    if img2text:
        tmp_x = x
        x = y
        y = tmp_x

    return dist(x, y), torch.arange(x.shape[0]).unsqueeze(0).expand(y.shape[0], -1).T.to(x.device), torch.eye(
        x.shape[0], dtype=bool).to(x.device)


def get_retrieval_metrics(x, y, dist = cosine_similarity_matrix, img2text = False):
    distances, indexes, gt = batched_matrix(x, y, dist, img2text=img2text)
    metrics = {
        'p@1': batched_top_k(distances, gt, k = 1),
        'p@5': batched_top_k(distances, gt, k = 5),
        'p@10': batched_top_k(distances, gt, k = 10)
    }

    return metrics


def rank_distance_to_self(distance_matrix):
    distances_from_rank_1 = []

    for i, row in enumerate(distance_matrix):
        # Rank the distances in the row (smaller distance is better, so argsort ascending)
        sorted_indices = np.argsort(row)  # Indices of sorted elements (ascending)

        # Find the rank of the diagonal element
        rank_of_diagonal = np.where(sorted_indices == i)[0][0] + 1  # Convert 0-indexed to rank

        # Distance from the first rank
        distance_from_rank_1 = rank_of_diagonal - 1
        distances_from_rank_1.append(distance_from_rank_1)

    return distances_from_rank_1

def eval_step(joint_model, dataloader, optimizer, loss_f, logger, epoch ):

    metrics = {
        'ranking_position': []

    }
    joint_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):

            scores = joint_model.retrieve(batch).detach().cpu().numpy()
            metrics['ranking_position'].extend(rank_distance_to_self(scores))





    metrics = {x: sum(y) / len(y) for x, y in zip(metrics, metrics.values())}
    return metrics