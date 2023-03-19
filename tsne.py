import numpy as np 
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


data_dir = '../datas/mnist/'
dataset = datasets.MNIST(data_dir, train=True, download=True)

idxs = np.arange(60000)
labels = dataset.train_labels.numpy()

# sort labels
idxs_labels = np.vstack((idxs, labels))
idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
idxs = idxs_labels[0, :]

idxs_train = [0,1,2,3,4,5,6,7,8,
              6001, 6002, 6003, 6004, 6005, 6006,6007, 6008,
              12003, 12004, 12005, 
              18004, 18005, 18006, 18007, 18008, 18009, 18010,
              24004, 24005,24006,24007,24008, 24009,
              30004, 30005,30006,
              36006,36007,36008, 36009,
              42007,42008,42009, 
              48007,48008, 48009,
              54001, 54002, 54008, 54009
              ]

idxs_train = idxs[idxs_train]
print((idxs_train))

datas = []
labels = []

for i in range(len(idxs_train)):
    image, label = dataset[idxs_train[i]]
    print(label)
    datas.append(np.array(image).flatten())
    labels.append(label)
np.save('50images2.npy', datas)
np.save('50labels.npy', labels)