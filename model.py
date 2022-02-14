import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregate(nn.Module):
    def __init__(self):
        super(Aggregate, self).__init__()

    def forward(self, x, A):
        if len(A.size()) > 2:
            x = torch.einsum('ncvl,nvw->ncwl', x, A)
        else:
            x = torch.einsum('ncvl,vw->ncwl', x, A)
        return x.contiguous()


class ConvFilter(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvFilter, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, K=2):
        super(GCN, self).__init__()
        self.agg = Aggregate()
        c_in = (K + 1) * c_in
        self.filter = ConvFilter(c_in, c_out)
        self.dropout = dropout
        self.K = K

    def forward(self, x, supports):
        out = [x]

        a = supports[0]
        x1 = self.agg(x, a)
        out.append(x1)
        for k in range(1, self.K):
            a = supports[k]
            x2 = self.agg(x1, a)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.filter(h)
        return h


class SST_GCN(nn.Module):
    def __init__(self, device, adj, dropout=0.3,
                 kernel_num=16, hist_size=4, kernel_size=2, layers=2, K=2):

        super(SST_GCN, self).__init__()
        self.dropout = dropout
        self.layers = layers
        self.hist_size = hist_size
        self.device = device
        self.tcn = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gcn = nn.ModuleList()
        self.start_conv = nn.Conv3d(in_channels=1,
                                    out_channels=kernel_num,
                                    kernel_size=(1, 1, 1),
                                    stride=(1, 1, 1),
                                    bias=False)

        self.adj = adj
        self.num_nodes = self.adj[0].size(0)
        self.zero_tensor = torch.tensor(0.0).to(self.device)
        self.one_tensor = torch.tensor(1.0).to(self.device)
        receptive_field = 1
        self.bias = []
        self.num_agg = K

        new_dilation = 1
        additional_scope = kernel_size - 1

        for b in range(layers):
            for i in range(self.hist_size):
                # dilated convolutions
                self.tcn.append(nn.Conv2d(in_channels=kernel_num,
                                          out_channels=kernel_num,
                                          kernel_size=(1, kernel_size),
                                          dilation=new_dilation,
                                          bias=False))

                self.gcn.append(GCN(kernel_num, kernel_num, dropout, self.num_agg))

            self.bn.append(nn.BatchNorm3d(kernel_num))
            self.bias.append(
                nn.Parameter(torch.ones(1, kernel_num, self.hist_size, 1, 1) * 0.01, requires_grad=True).to(device))

            new_dilation *= 2
            receptive_field += additional_scope
            additional_scope *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=kernel_num * self.hist_size,
                                    out_channels=kernel_num * self.hist_size // 2,
                                    kernel_size=(1,),
                                    bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=kernel_num * self.hist_size // 2,
                                    out_channels=self.hist_size,
                                    kernel_size=(1,),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input, context):

        x = torch.unsqueeze(input, dim=1)
        context = context[1]
        if len(context.size()) > 2:
            context = context[:, 3, :]

        x = self.start_conv(x)
        x = torch.transpose(x, 2, 4)
        agg_1 = self.context_diffusion(context)
        for l in range(self.layers):
            aggregations = [agg_1]
            for k in range(1, self.num_agg):
                diag = torch.diagonal(agg_1, dim1=1, dim2=2)
                new_ctx = torch.where(diag == 1.0, self.zero_tensor, self.one_tensor)
                agg_2 = self.context_diffusion(new_ctx)
                aggregations.append(agg_2)
                agg_1 = agg_2

            residual = x
            tensor_list = []
            for i in range(self.hist_size):
                hist_i = x[:, :, i, :, :]
                filter = self.tcn[l * self.hist_size + i](hist_i)
                hist_i = F.relu(filter)
                hist_i = self.gcn[l * self.hist_size + i](hist_i, aggregations)
                tensor_list.append(torch.unsqueeze(hist_i, dim=2))

            x = torch.cat(tensor_list, dim=2)
            x = x + residual[..., -x.size(-1):]
            x = F.relu(x + self.bias[l])
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.bn[l](x)

        x = torch.squeeze(x, dim=-1)

        x = torch.reshape(x, (x.size(0), -1, x.size(-1)))
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x.transpose(1, 2)
        x = torch.softmax(x, dim=2)

        return x

    def context_diffusion(self, ctx):
        ctx = torch.unsqueeze(ctx, dim=-1)
        mask_adj = torch.einsum('nmv,mc->nmc', ctx, self.adj[0]) + torch.eye(self.num_nodes).to(self.device)
        mask_adj = torch.where(mask_adj > 0, self.one_tensor, self.zero_tensor.to(self.device))
        rowsum = torch.sum(mask_adj, dim=1)
        d_inv = torch.pow(rowsum, -1)
        d_mat = torch.diag_embed(d_inv)
        diff_adj = torch.matmul(mask_adj, d_mat)
        return diff_adj