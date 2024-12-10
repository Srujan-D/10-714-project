import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset
from dgl import DGLGraph
from dgl.data import citegrh
import dgl

class CoraDataset(Dataset):
    def __init__(self, root='~/.dgl', name='cora', data_type='train'):
        super(CoraDataset, self).__init__(root)
        self.name = name

        self.download()
        self.features = self.data.features.astype(np.float32)
        self.ids = np.arange(self.features.shape[0], dtype=np.int64)
        graph = dgl.transform.add_self_loop(DGLGraph(self.data.graph))
        self.src, self.dst = graph.edges()
        self.labels = self.data.labels.astype(np.int64)
        self.adj = graph.adjacency_matrix().to_dense().numpy()

        if data_type == 'train':
            self.mask = self.data.train_mask.astype(bool)
        elif data_type == 'val':
            self.mask = self.data.val_mask.astype(bool)
        elif data_type == 'test':
            self.mask = self.data.test_mask.astype(bool)
        print('{} Dataset for {} Loaded.'.format(self.name, data_type))

    def __len__(self):
        return np.sum(self.mask)

    def __getitem__(self, index):
        masked_ids = self.ids[self.mask]
        index_id = masked_ids[index]
        neighbors = self.features[self.dst[self.src == index_id]]
        return (
            np.expand_dims(self.features[index_id], axis=0),
            self.labels[index_id],
            np.expand_dims(neighbors, axis=0)
        )

    def download(self):
        print('Loading {} Dataset...'.format(self.name))
        processed_folder = os.path.join(self.root, self.name)
        os.makedirs(processed_folder, exist_ok=True)
        os.environ["DGL_DOWNLOAD_DIR"] = processed_folder
        data_file = os.path.join(processed_folder, 'data.pkl')

        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = citegrh.load_cora()
            with open(data_file, 'wb') as f:
                pickle.dump(self.data, f)

        self.feat_len, self.num_class = self.data.features.shape[1], self.data.num_labels
