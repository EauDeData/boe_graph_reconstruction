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

def eval_step(joint_model, dataloader, optimizer, loss_f, logger, epoch ):

    metrics = {
        'mAP': []

    }
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):

            node_features = joint_model(batch)
            scores = cosine_similarity_matrix(node_features['queries'], node_features['document'])

            for idx_batch in range(scores.shape[0]):
                metrics['mAP'].append(average_precision_score(y_true=np.array([i == idx_batch for i in batch['batch_indices']]),
                                             y_score=scores[idx_batch].cpu().numpy()
                                             ))



    metrics = {x: sum(y) / len(y) for x, y in zip(metrics, metrics.values())}
    return metrics