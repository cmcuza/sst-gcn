import pickle
import numpy as np
import os
import torch
import scipy.sparse as sp
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, ws, ctx, batch_size):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ws = ws
        self.ctx = ctx

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, ws, ctx = self.xs[permutation], self.ys[permutation], self.ws[permutation], self.ctx[permutation]
        self.xs = xs
        self.ys = ys
        self.ws = ws
        self.ctx = ctx

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                w_i = self.ws[start_ind: end_ind, ...]
                c_i = [self.ctx[0][start_ind: end_ind, ...], self.ctx[1][start_ind: end_ind, ...]]
                yield x_i, y_i, w_i, c_i

                self.current_ind += 1

        return _wrapper()


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    adj_mx = adj_mx + sp.identity(adj_mx.shape[0])
    adj = [sym_adj(adj_mx)]
    return adj


def load_dataset(args):
    data = {}
    validation_size = args.batch_size * 5
    ctx_base = os.path.join(args.data, 'context')
    for category in ['train', 'test']:
        base = os.path.join(args.data, category)
        data['x_' + category] = load_pickle(os.path.join(base, f'X_{category}_{args.fold}.pickle'))
        data['y_' + category] = load_pickle(os.path.join(base, f'Y_{category}_{args.fold}.pickle'))
        data['w_' + category] = load_pickle(os.path.join(base, f'weight_matrix_{args.fold}.pickle'))
        data['ctx_' + category] = load_pickle(os.path.join(ctx_base, f'X_{category}_ctx_{args.fold}.pickle'))

    validation_index = np.random.RandomState(42).choice(data['x_train'].shape[0], validation_size, replace=False)
    data['x_val'] = data['x_train'][validation_index, ...]
    data['y_val'] = data['y_train'][validation_index, ...]
    data['w_val'] = data['w_train'][validation_index, ...]
    data['ctx_val'] = [data['ctx_train'][0][validation_index, ...], data['ctx_train'][1][validation_index, ...]]

    train_index = np.setdiff1d(np.arange(data['x_train'].shape[0]), validation_index)
    data['x_train'] = data['x_train'][train_index, ...]
    data['y_train'] = data['y_train'][train_index, ...]
    data['w_train'] = data['w_train'][train_index, ...]
    data['ctx_train'] = [data['ctx_train'][0][train_index, :], data['ctx_train'][1][train_index, :]]

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], data['w_train'], data['ctx_train'], args.batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], data['w_val'], data['ctx_val'], args.batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], data['w_test'], data['ctx_test'], args.batch_size)

    return data


def load_test(directory, batch_size):
    data = {}
    category = 'test'
    base = os.path.join(directory, category)
    ctx_base = os.path.join(directory, 'context')
    data['x_' + category] = load_pickle(os.path.join(base, f'X_{category}.pickle'))
    data['y_' + category] = load_pickle(os.path.join(base, f'Y_{category}.pickle'))
    data['w_' + category] = load_pickle(os.path.join(base, f'weight_matrix.pickle'))
    data['ctx_' + category] = load_pickle(os.path.join(ctx_base, f'X_{category}_ctx.pickle'))
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], data['w_test'], data['ctx_test'], batch_size)

    return data

def weighted_kl(y_pred, y_true, weight, epsilon=1e-3):
    N, M, B = y_pred.size()
    w_N, w_M = weight.size()
    assert w_N == N, w_M == M

    log_pred = torch.log10(y_pred + epsilon)
    log_true = torch.log10(y_true + epsilon)
    log_sub = torch.subtract(log_pred, log_true)
    mul_op = torch.multiply(y_pred, log_sub)
    sum_hist = torch.sum(mul_op, dim=2)
    if weight is not None:
        sum_hist = torch.multiply(weight, sum_hist)
        #        avg_kl_div = tf.reduce_mean(sum_hist)
    weight_avg_kl_div = torch.sum(sum_hist)
    avg_kl_div = weight_avg_kl_div / torch.sum(weight)

    return avg_kl_div

