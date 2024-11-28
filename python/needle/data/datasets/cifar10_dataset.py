import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        super().__init__(transforms)
        self.base_folder = base_folder
        self.train = train
        self.p = p
        self.X = None
        self.y = None

        if self.train:
            self.X = np.zeros((50000, 3, 32, 32))
            self.y = np.zeros(50000)
            for i in range(5):
                with open(os.path.join(self.base_folder, f'data_batch_{i+1}'), 'rb') as f:
                    data = pickle.load(f, encoding='bytes')
                    self.X[i*10000:(i+1)*10000] = data[b'data'].reshape(10000, 3, 32, 32)
                    self.y[i*10000:(i+1)*10000] = data[b'labels']
        else:
            with open(os.path.join(self.base_folder, 'test_batch'), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                self.X = data[b'data'].reshape(10000, 3, 32, 32)
                self.y = data[b'labels']
                self.y = np.array(self.y)
        self.X = self.X / 255       

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.transforms:
            X_items = np.array([self.apply_transforms(x) for x in self.X[index]])
        else:
            X_items = np.array(self.X[index])
        return X_items, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return len(self.X)
        ### END YOUR SOLUTION
