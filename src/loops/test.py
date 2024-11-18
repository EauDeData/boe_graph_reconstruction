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

def eval_step(joint_model, dataloader, optimizer, loss_f, logger, epoch, single_batch = False):

    metrics = {key: {'acc@10': [], 'acc@5': [], 'acc@1': [], 'ranking_position': []} for key in ['doc2query', 'query2doc']}

    joint_model.eval()
    loss_buffer = []

    skipped = []
    with torch.no_grad():
        for i, big_batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            loss_buffer.append(joint_model(big_batch).item())

            documents = []
            queries = []

            (dense_doc_features, doc_features_mask), (dense_query_features, query_features_mask), _ = joint_model.common_process(big_batch)
            if single_batch:
                dense_doc_features = dense_doc_features.squeeze()[None, :]
                dense_query_features = dense_query_features.squeeze()[None, :]

            if len(dense_doc_features.shape) == 2 or len(dense_query_features.shape) == 2: continue

            # print(big_batch['idx_images_document'], doc_features_mask)
            t1_edges = joint_model.hungarian_distance.produce_reading_order_edges(dense_doc_features.shape[1]).to(joint_model.device)
            t2_edges = joint_model.hungarian_distance.produce_reading_order_edges(dense_query_features.shape[1]).to(joint_model.device)
            message_passed_t1 = [joint_model.hungarian_distance.gat(
                x=dense_doc_features[i],
                edge_index=t1_edges
            ) for i in range(dense_doc_features.shape[0])]

            message_passed_t2 = [joint_model.hungarian_distance.gat(
                x=dense_query_features[i],
                edge_index=t2_edges
            ) for i in range(dense_query_features.shape[0])]

            documents.extend([{'document_features': dense_doc_features[i][doc_features_mask[i]], 'context_doc_features': message_passed_t1[i][doc_features_mask[i]]} for i in range(dense_doc_features.shape[0])])
            queries.extend([{'query_features': dense_query_features[i][query_features_mask[i]], 'context_query_features': message_passed_t2[i][query_features_mask[i]]} for i in range(dense_query_features.shape[0])])

            for i, (query, document) in enumerate(zip(queries, documents)):

                # Lists to store distances for current query/document pair
                doc2query_dists = []
                query2doc_dists = []

                # Loop over the entire set to compute distances
                for j, (query_j, document_j) in enumerate(zip(queries, documents)):
                    # Document to Query distance
                    doc2query_d = joint_model.hungarian_distance.hungarian_distance(
                        document['document_features'], query_j['query_features'],
                        document['context_doc_features'], query_j['context_query_features']
                    )
                    doc2query_dists.append(doc2query_d.detach().cpu())

                    # Query to Document distance
                    query2doc_d = joint_model.hungarian_distance.hungarian_distance(
                        document_j['document_features'], query['query_features'],
                        document_j['context_doc_features'], query['context_query_features']
                    )
                    query2doc_dists.append(query2doc_d.detach().cpu())

                # Sort distances and get indices (for ranking purposes)
                doc2query_sorted_idx = np.argsort(doc2query_dists)  # Ascending order
                query2doc_sorted_idx = np.argsort(query2doc_dists)

                # Find the position (rank) of the current document/query
                doc2query_rank = np.where(doc2query_sorted_idx == i)[0][0]  # Rank of the current document in doc2query
                query2doc_rank = np.where(query2doc_sorted_idx == i)[0][0]  # Rank of the current query in query2doc

                # Update doc2query metrics
                metrics['doc2query']['acc@1'].append(1 if doc2query_rank < 1 else 0)
                metrics['doc2query']['acc@5'].append(1 if doc2query_rank < 5 else 0)
                metrics['doc2query']['acc@10'].append(1 if doc2query_rank < 10 else 0)
                metrics['doc2query']['ranking_position'].append(doc2query_rank)

                # Update query2doc metrics
                metrics['query2doc']['acc@1'].append(1 if query2doc_rank < 1 else 0)
                metrics['query2doc']['acc@5'].append(1 if query2doc_rank < 5 else 0)
                metrics['query2doc']['acc@10'].append(1 if query2doc_rank < 10 else 0)
                metrics['query2doc']['ranking_position'].append(query2doc_rank)


    for key in metrics:
        for k in metrics[key]:
            metrics[key][k] = np.mean(metrics[key][k])
    metrics['evaluation_loss'] = sum(loss_buffer) / len(loss_buffer)
    return metrics