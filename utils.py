import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import re
import torch
from time import perf_counter
import tabulate
import math
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def top_k_edge_weight_sampling(adj, k):
    """
    Select the top k neighbors for each node based on the largest edge weights.

    :param adj: scipy.sparse matrix representing the weighted adjacency matrix.
    :param k: The number of neighbors to select for each node based on the largest edge weights.
    :return: A new adjacency matrix with top k selection based on edge weights.
    """
    if not isinstance(adj, sp.csr_matrix):
        adj = adj.tocsr()

    new_adj = sp.lil_matrix(adj.shape)

    for i in range(adj.shape[0]):
        neighbors = adj[i].nonzero()[1]
        edge_weights = adj[i].data

        if len(neighbors) > k:
            # Selecting top k neighbors based on highest edge weights
            top_k_indices = np.argsort(-edge_weights)[:k]  # Negate for descending order
            chosen_neighbors = neighbors[top_k_indices]
            new_adj[i, chosen_neighbors] = adj[i, chosen_neighbors]

    return new_adj.tocsr()


def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    index_dict = {}
    label_dict = {}
    phases = ["train", "val", "test"]
    objects = []
    def load_pkl(path):
        with open(path.format(dataset_str, p), 'rb') as f:
            if sys.version_info > (3, 0):
                return pkl.load(f, encoding='latin1')
            else:
                return pkl.load(f)

    for p in phases:
        index_dict[p] = load_pkl("data/ind.{}.{}.x".format(dataset_str, p))
        label_dict[p] = load_pkl("data/ind.{}.{}.y".format(dataset_str, p))

    adj = load_pkl("data/ind.{}.BCD.adj".format(dataset_str))
    adj = adj.astype(np.float32)
    max_value = adj.max()
    min_value = adj[adj.nonzero()].min()
    print("max"  + str(max_value) +  " min " + str(min_value))
    adj = preprocess_adj(adj)

    return adj, index_dict, label_dict

def k_neighbour_sampling(adj, k):
    """
    Perform k-neighbour sampling on the adjacency matrix.

    :param adj: scipy.sparse matrix representing the adjacency matrix.
    :param k: The number of neighbors to sample for each node.
    :return: A new adjacency matrix with k-neighbour sampling applied.
    """
    if not isinstance(adj, sp.csr_matrix):
        adj = adj.tocsr()

    # Limit the number of neighbors to k
    for i in range(adj.shape[0]):
        row = adj[i]
        if row.nnz > k:
            # Choose k neighbors randomly
            indices = np.random.choice(row.indices, k, replace=False)
            new_values = np.ones_like(indices, dtype=np.float32)
            adj[i] = sp.csr_matrix((new_values, (np.zeros(k), indices)), shape=row.shape)

    return adj


def normalize_laplacian(adj):
    """Compute the normalized Laplacian matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).transpose().tocoo()

# def preprocess_adj(adj, max_len):
#     """Preprocessing of adjacency matrix with k-neighbour sampling for GCN model."""
#     # adj_sampled = k_neighbour_sampling(adj, math.floor(aver_len))
#     adj_sampled = top_k_edge_weight_sampling(adj, max_len)
#     # adj_sampled = edge_weight_importance_sampling(adj, max_len)
#     adj_normalized = normalize_adj(adj_sampled + sp.eye(adj_sampled.shape[0]))
#     return adj_normalized

# def compute_importance(adj):
#     """ Compute importance scores based on node degree. """
#     importance_scores = np.array(adj.sum(axis=1)).flatten()
#     return importance_scores

# def importance_sampling(adj, k):
#     """
#     Perform importance sampling on the adjacency matrix.

#     :param adj: scipy.sparse matrix representing the adjacency matrix.
#     :param k: The number of neighbors to sample for each node.
#     :return: A new adjacency matrix with importance sampling applied.
#     """
#     if not isinstance(adj, sp.csr_matrix):
#         adj = adj.tocsr()

#     importance_scores = compute_importance(adj)
#     new_adj = sp.lil_matrix(adj.shape)

#     for i in range(adj.shape[0]):
#         neighbors = adj[i].nonzero()[1]
#         if len(neighbors) > k:
#             # Sampling based on importance
#             probabilities = importance_scores[neighbors]
#             probabilities = probabilities / probabilities.sum()
#             chosen_neighbors = np.random.choice(neighbors, k, replace=False, p=probabilities)
#             new_adj[i, chosen_neighbors] = 1

#     return new_adj.tocsr()

# def edge_weight_importance_sampling(adj, k):
#     """
#     Perform importance sampling on the adjacency matrix based on edge weights.

#     :param adj: scipy.sparse matrix representing the weighted adjacency matrix.
#     :param k: The number of neighbors to sample for each node.
#     :return: A new adjacency matrix with importance sampling applied.
#     """
#     if not isinstance(adj, sp.csr_matrix):
#         adj = adj.tocsr()

#     new_adj = sp.lil_matrix(adj.shape)

#     for i in range(adj.shape[0]):
#         neighbors = adj[i].nonzero()[1]
#         edge_weights = adj[i].data

#         if len(neighbors) > k:
#             # Sampling based on edge weights
#             total_weight = edge_weights.sum()
#             probabilities = edge_weights / total_weight
#             chosen_neighbors = np.random.choice(neighbors, k, replace=False, p=probabilities)
#             new_adj[i, chosen_neighbors] = adj[i, chosen_neighbors]

#     return new_adj.tocsr()


# def preprocess_adj(adj, k):
    
#     """Preprocessing of adjacency matrix for simple GCN model using the normalized Laplacian."""
#     adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
#     return adj_normalized
#     # laplacian = normalize_laplacian(adj)
#     # return laplacian

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    string = re.sub(r'[?|$|.|!]',r'',string)
    string = re.sub(r'[^a-zA-Z0-9 ]',r'',string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def sparse_to_torch_sparse(sparse_mx, device='cpu'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    if device == 'cuda':
        indices = indices.cuda()
        values = torch.from_numpy(sparse_mx.data).cuda()
        shape = torch.Size(sparse_mx.shape)
        adj = torch.cuda.sparse.FloatTensor(indices, values, shape)
    elif device == 'cpu':
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        adj = torch.sparse.FloatTensor(indices, values, shape)
    return adj

def sparse_to_torch_dense(sparse, device='cpu'):
    dense = sparse.todense().astype(np.float32)
    torch_dense = torch.from_numpy(dense).to(device=device)
    return torch_dense

def sgc_precompute(adj, features, degree, index_dict):
    assert degree==1, "Only supporting degree 2 now"
    feat_dict = {}
    start = perf_counter()
    train_feats = features[:, index_dict["train"]]
    train_feats = torch.spmm(adj, train_feats).t()
    train_feats_max, _ = train_feats.max(dim=0, keepdim=True)
    train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    train_feats_range = train_feats_max-train_feats_min
    useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
    train_feats = train_feats[:, useful_features_dim]
    train_feats_range = train_feats_range[:, useful_features_dim]
    train_feats_min = train_feats_min[:, useful_features_dim]
    train_feats = (train_feats-train_feats_min)/train_feats_range
    feat_dict["train"] = train_feats
    for phase in ["test", "val"]:
        feats = features[:, index_dict[phase]]
        feats = torch.spmm(adj, feats).t()
        feats = feats[:, useful_features_dim]
        feat_dict[phase] = ((feats-train_feats_min)/train_feats_range).cpu() # adj is symmetric!
    precompute_time = perf_counter()-start
    return feat_dict, precompute_time

def set_seed(seed, cuda= False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def print_table(values, columns, epoch):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
