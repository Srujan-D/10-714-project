"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from models import *
import time
import os
import numpy as np
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
from models import GCN

device = ndl.cuda()

def cora_data(root='~/.dgl', name='cora', device="cpu"):
    name = name.lower()
    print("Loading %s Dataset" % (name))
    processed_folder = os.path.join(root, name.lower())
    os.makedirs(processed_folder, exist_ok=True)
    os.environ["DGL_DOWNLOAD_DIR"] = processed_folder

    data = citegrh.load_cora()

    # Convert data to numpy arrays
    features = np.array(data.features, dtype=np.float32)
    graph = dgl.transform.add_self_loop(DGLGraph(data.graph))
    adj = graph.adjacency_matrix().to_dense().numpy()
    adj = adj.astype(np.float32)
    labels = np.array(data.labels, dtype=np.float32)

    idx_train = np.array(data.train_mask, dtype=np.float32)
    idx_val = np.array(data.val_mask, dtype=np.float32)
    idx_test = np.array(data.test_mask, dtype=np.float32)

    feat_len, num_class = features.shape[1], data.num_labels
    return adj, features, labels, idx_train, idx_val, idx_test, feat_len, num_class

def accuracy(output, labels):
    output = output.data.numpy()
    labels = labels.data.numpy()


    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


### CIFAR-10 training ###
def epoch_general_cora(model, data,  loss_fn=nn.SoftmaxLoss(), opt=None):

    adj, features, labels, idx_train, idx_val, idx_test, feat_len, num_class = data

    adj = ndl.SparseTensor(adj, device=device)    #TODO: change to SparseTensor
    features = ndl.Tensor(features, device=device)
    labels = ndl.Tensor(labels, device=device)

    # idx_train = ndl.Tensor(idx_train, device=device)
    # idx_val = ndl.Tensor(idx_val, device=device)
    idx_test = ndl.Tensor(idx_test, device=device)

    if opt is None:
        model.eval()
        output = model(features, adj)
        loss_test = loss_fn(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return loss_test, acc_test
    else:
        model.train()
        model.train()
        opt.reset_grad()
        
        output = model(features, adj)
        output_list = list(ndl.ops.split(output, 0))
        labels_list = list(ndl.ops.split(labels, 0))

        losses = []
        # accuracy_list = []
        for index, i in enumerate(idx_train):
            if i == 1.:
                # breakpoint()
                loss = loss_fn(output_list[index], labels_list[index])
                losses.append(loss)

                # accuracy_list.append(accuracy(output_list[index], labels_list[index]))

        # breakpoint()
        loss_train = ndl.ops.stack(losses, 0)
        loss_train = ndl.ops.summation(loss_train)

        # accuracy_train = ndl.ops.stack(accuracy_list, 0)
        # accuracy_train = ndl.ops.summation(accuracy_train)               
        

        # loss_train = loss_fn(output[idx_train], labels[idx_train])
        # acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        opt.step()
        # loss_val = loss_fn(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        return 0.1, loss_train
    ### END YOUR SOLUTION


def train_cora(
    n_epochs=3,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    loss_fn=nn.SoftmaxLoss(),
):
    np.random.seed(4)
    
    data = cora_data()
    adj, features, labels, idx_train, idx_val, idx_test, feat_len, num_class = data

    model = GCN(nfeat=feat_len, nhid=16, nclass=num_class, dropout=0.5, device=device)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cora(model,data, loss_fn=loss_fn, opt=opt)
        print(f"Epoch: {epoch}, Acc: {avg_acc}, Loss: {avg_loss}")

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
