import argparse
import os.path as p

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str,default='cpu', help='cpu if running on CPU or cuda if running on GPU')
parser.add_argument('--data', type=str, default=p.join('data', 'rm_0.5', 'chengdu173'), help='data path')
parser.add_argument('--adjdata', type=str,default=p.join('data', 'rm_0.5', 'chengdu173', 'edge_adj.pickle'), help='adj data path')
parser.add_argument('--nhid', type=int, default=32, help='Size of hidden units')
parser.add_argument('--hist_size', type=int, default=4, help='inputs dimension or histograms')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--print_every', type=int, default=50, help='Logging')
parser.add_argument('--save', type=str,default=p.join('garage', 'chengdu40'), help='save path')
parser.add_argument('--fold', type=int, default=0, help='current fold')
parser.add_argument('--hist', default=[10, 20, 30, 40])

