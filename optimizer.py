import torch.optim as optim
from model import *
import util


class StochasticGD:
    def __init__(self, nhid, dropout, lrate, wdecay, device, adj):
        self.model = SST_GCN(device, adj, dropout=dropout, kernel_num=nhid)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.weighted_kl
        self.clip = 5

    def train(self, input, real, weights, ctx):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, ctx)
        loss = self.loss(output, real, weights)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()

    def eval(self, input, real, weights, ctx):
        self.model.eval()
        output = self.model(input, ctx)
        loss = self.loss(output, real, weights)

        return loss.item()
