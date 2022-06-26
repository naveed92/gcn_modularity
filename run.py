"""Adapted from: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py"""

import argparse
import torch
import numpy as np
from src.utils import preprocess_adj, preprocess_features, sparse_mx_to_torch_sparse_tensor, load_data, accuracy
import torch.nn.functional as F
import time
from src.models import GCN, GCNMod, GCNAux
from torch import optim

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    help='Specify dataset from values: cora, citeseer, pubmed')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--n_label_per_class', type=int, default=20,
                    help='Number of positive samples per class')
parser.add_argument('--weight', type=float, default=0.0,
                    help='Weight of modularity loss')
"""
parser.add_argument('--model_type', type=str, default='gcn',
                    help='Select model type from: gcn, gcn_mod, gcn_aux')
"""

args = parser.parse_args()

if args.weight == 0.0:
    args.model_type = 'gcn'
else:
    args.model_type == 'gcn_mod'

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = args.dataset # cora, citeseer or pubmed

adj, features, labels, idx_train, idx_test, idx_val = load_data(dataset)

idx_train, idx_test, idx_val = torch.LongTensor(idx_train), torch.LongTensor(idx_test), torch.LongTensor(idx_val)
features = torch.FloatTensor(preprocess_features(features).toarray())

n_nodes = adj.shape[1]
n_features = features.shape[1]
n_class = labels.shape[1]

adj_norm = sparse_mx_to_torch_sparse_tensor(preprocess_adj(adj))
labels = torch.LongTensor([np.where(l)[0][0] if sum(l) > 0 else -1 for l in labels.toarray()])

label_to_idx = {}
for i in range(n_class):
    idx = np.where(labels==i)[0]
    label_to_idx[i] = idx

idx_train = np.concatenate([np.random.choice(label_to_idx[i], size=args.n_label_per_class, replace=False) 
                            for i in range(n_class)])
idx_train.sort()

if args.model_type == 'gcn':
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=n_class,
                dropout=args.dropout)

elif args.model_type == 'gcn_mod':
    model = GCNMod(nfeat=features.shape[1],
               nhid=args.hidden,
               nclass=n_class,
               dropout=args.dropout,
               adj=adj,
               weight=args.weight)
               
elif args.model_type == 'gcn_aux':
    model = GCNAux(nfeat=features.shape[1],
                   nhid=args.hidden,
                   nclass=n_class,
                   dropout=args.dropout,
                   adj=adj,
                   weight=args.weight)
    
else:
    model = None
               
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
               
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj_norm = adj_norm.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    output = model(features, adj_norm)
    
    if args.model_type == 'gcn_aux':
        output_aux = model.forward_aux(features, adj_norm)
        loss_train = model.loss(output, output_aux, labels, idx_train)
    else:
        output_aux = None
        loss_train = model.loss(output, labels, idx_train)

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately, deactivates dropout during validation run.
        model.eval()
        output = model(features, adj_norm)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj_norm)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()