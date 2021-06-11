import numpy as np
import os
import torch
import pickle as pkl
from scipy.stats import wasserstein_distance
from pyemd import emd


def masked_mse(preds, labels, weights):
    loss = (preds - labels) ** 2
    loss = loss * weights
    return torch.sum(loss)/torch.sum(weights)


def masked_rmse(preds, labels, weights):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, weights=weights))


def masked_mae(preds, labels, weights):
    loss = torch.abs(preds - labels)
    loss = loss * weights
    return torch.sum(loss)/torch.sum(weights)


def masked_mape(preds, labels, weights):
    loss = torch.abs(preds - labels) / (labels)
    loss = loss * weights
    return torch.sum(loss)/torch.sum(weights)


def metric(pred, real, weigth):
    mae = masked_mae(pred, real, weigth).item()
    mape = masked_mape(pred, real, weigth).item()
    rmse = masked_rmse(pred, real, weigth).item()
    return mae, mape, rmse


class Evaluator:
    def __init__(self, args):
        all_hist_dir = os.path.join(args.data, "hist/hist_all_{}.pickle".format(args.fold))
        self.EPSILON = 1e-3
        with open(all_hist_dir, 'rb') as f:
            self.all_hist = pkl.load(f)
        #self.all_hist = self.all_hist.apply(funct)
        self.all_hist = self.all_hist[:].values[0].tolist()
        self.all_hist = np.asarray([e.tolist() for e in self.all_hist])
        self.device = args.device

    def mklr(self, y_true, y_pred, weights):
        ha = torch.Tensor(np.asarray([self.all_hist] * y_true.shape[0])).to(self.device)
        KL_num = self.kl(y_true, y_pred)
        KL_den = self.kl(y_true, ha)

        L_ha = torch.multiply(weights, KL_den)
        scalar_ha = torch.sum(L_ha)
        L_den = scalar_ha  # /scaler
        L_pred = torch.multiply(weights, KL_num)
        scalar_pred = torch.sum(L_pred)
        L_num = scalar_pred  # /scaler

        return L_num/L_den

    def jesen_shannon_divergence(self, y_true, y_pred, weights):
        m_hat = 0.5*(y_true+y_pred)
        kl1 = self.kl(y_true, m_hat)
        kl2 = self.kl(y_pred, m_hat)
        js = 0.5*(kl1+kl2)
        wjs = torch.multiply(weights, js)
        scaler = torch.sum(weights)

        return torch.sum(wjs) / scaler

    def kl(self, y_true, y_pred):
        numerator = torch.log10(y_pred + self.EPSILON)
        denominator = torch.log10(y_true + self.EPSILON)
        log_ratio = torch.subtract(numerator, denominator)
        prob_log = torch.multiply(y_pred, log_ratio)
        kl = torch.sum(prob_log, dim=2)
        return kl

    def wasserstein(self, y_true, y_pred, weights):
        ytrue = np.array(y_true)
        ypred = np.array(y_pred)
        n = ypred.shape[0]
        m = ypred.shape[1]
        earth_dist = torch.tensor([wasserstein_distance(ytrue[i, j, :], ypred[i, j, :]) for i in range(n) for j in range(m)]).reshape((n, m))
        masked_dist = np.multiply(weights, earth_dist)
        distance = torch.sum(masked_dist)/torch.sum(weights)
        return distance

    def weighted_emd(self, y_true, y_pred, weigths):
        y_true = np.array(y_true.to('cpu')).astype(np.float64)
        y_pred = np.array(y_pred.to('cpu')).astype(np.float64)

        b = y_pred.shape[2]
        n = y_pred.shape[0]
        m = y_pred.shape[1]

        d_matrix = np.zeros((b, b))
        for i in range(b):
            d_matrix[i, i:b] = np.arange(b)[:b - i]

        d_matrix = np.maximum(d_matrix, d_matrix.T)
        earth_dist = torch.tensor(
            [emd(y_true[i, j, :], y_pred[i, j, :], d_matrix) for i in range(n) for j in range(m)]).reshape((n, m))

        masked_emd = torch.multiply(weigths, earth_dist)
        distance = torch.sum(masked_emd) / torch.sum(weigths)

        return distance

    def hist_metrics(self, y_true, y_pred, weights):
        n, m, b = y_pred.size()
        mae_tensor = torch.empty(size=(b,))
        rmse_tensor = torch.empty(size=(b,))
        mape_tensor = torch.empty(size=(b,))
        for i in range(b):
            true_hist_i = y_true[..., i]
            pred_hist_i = y_pred[..., i]
            mae = masked_mae(pred_hist_i, true_hist_i, weights)
            rmse = masked_rmse(pred_hist_i, true_hist_i, weights)
            mape = masked_mape(pred_hist_i, true_hist_i, weights)
            rmse_tensor[i] = rmse
            mae_tensor[i] = mae
            mape_tensor[i] = mape

        mean_mae = torch.mean(mae_tensor)
        #mean_mape = torch.mean(mape_tensor)
        mean_rmse = torch.mean(rmse_tensor)

        return mean_mae, mean_rmse  # mean_mape,