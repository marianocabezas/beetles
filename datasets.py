import os
import cv2
import numpy as np
from copy import deepcopy
from torch.utils.data.dataset import Dataset


class BalancedDataset(Dataset):
    """
    Dataset that loads images stored in different folders based on their class.
    Large classes are undersampled according to the samples of the smallest class
    (per epoch). However, repetitions are not allowed until the whole large class
    has been observed.
    """

    def __init__(self, path, classes):
        self.min_len = np.inf
        self.classes = classes
        data = []
        for label in classes:
            label_path = os.path.join(path, label)
            label_data = []
            files = sorted(os.listdir(label_path))
            if len(files) < self.min_len:
                self.min_len = len(files)
            for patch_name in files:
                patch_path = os.path.join(label_path, patch_name)
                patch = np.moveaxis(
                    cv2.imread(patch_path)[1:-1, 1:-1, ::-1], -1, 0
                )
                label_data.append(patch)
            data.append(label_data)

        self.data = data
        self.permuted = deepcopy(self.data)

    def __getitem__(self, index):
        k = index // self.min_len
        if len(self.data[k]) == self.min_len:
            patch_index = index % self.min_len
            x = self.data[k][patch_index]
            print(self.classes[k])
        else:
            print(self.classes[k], len(self.permuted[k]))
            patch_index = np.random.randint(
                0, len(self.permuted[k])
            )
            x = self.permuted[k].pop(patch_index)
            print(self.classes[k], len(self.permuted[k]), patch_index)
            if len(self.permuted[k]) == 0:
                self.permuted[k] = deepcopy(self.data[k])

        return x, k

    def __len__(self):
        return self.min_len * len(self.classes)


class NaturalDataset(Dataset):
    """
    Dataset that loads images stored in different folders based on their class.
    """

    def __init__(self, path, classes, shuffle=True):
        self.min_len = np.inf
        self.classes = classes
        data = []
        labels = []
        for k, label in enumerate(classes):
            label_path = os.path.join(path, label)
            files = sorted(os.listdir(label_path))
            for patch_name in files:
                patch_path = os.path.join(label_path, patch_name)
                patch = np.moveaxis(
                    cv2.imread(patch_path)[1:-1, 1:-1, ::-1], -1, 0
                )
                data.append(patch)
                labels.append(k)

        if shuffle:
            permutation = np.random.permutation(len(data))
            self.data = np.array(data)[permutation]
            self.labels = np.array(labels)[permutation]
        else:
            self.data = np.array(data)
            self.labels = np.array(labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return x, y

    def __len__(self):
        return len(self.data)
