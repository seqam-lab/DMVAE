import os
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
import torch
from torchnet.dataset.dataset import Dataset
from torchvision import datasets, transforms


'''
 Data and code from https://github.com/iffsid/mmvae
'''
class ResampleDataset(Dataset):

    def __init__(self, dataset, sampler=lambda ds, idx: idx, size=None):
        super(ResampleDataset, self).__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.size = size

    def __len__(self):
        return (self.size and self.size > 0) and self.size or len(self.dataset)

    def __getitem__(self, idx):
        super(ResampleDataset, self).__getitem__(idx)
        idx = self.sampler(self.dataset, idx)

        return self.dataset[idx]


def getPairedDataset(path, batch_size, shuffle=True, cuda=True):
    if not (os.path.exists(path + '/train-ms-mnist-idx.pt')
            and os.path.exists(path + '/train-ms-svhn-idx.pt')
            and os.path.exists(path + '/test-ms-mnist-idx.pt')
            and os.path.exists(path + '/test-ms-svhn-idx.pt')):
        raise RuntimeError('Generate transformed indices with the script in bin')
    # get transformed indices
    t_mnist = torch.load(path + '/train-ms-mnist-idx.pt')
    t_svhn = torch.load(path + '/train-ms-svhn-idx.pt')
    s_mnist = torch.load(path + '/test-ms-mnist-idx.pt')
    s_svhn = torch.load(path + '/test-ms-svhn-idx.pt')

    # load base datasets

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    tx = transforms.ToTensor()
    t1 = DataLoader(datasets.MNIST(path, train=True, download=True, transform=tx),
                       batch_size=batch_size, shuffle=shuffle, **kwargs)
    s1 = DataLoader(datasets.MNIST(path, train=False, download=True, transform=tx),
                      batch_size=batch_size, shuffle=shuffle, **kwargs)

    t2 = DataLoader(datasets.SVHN(path, split='train', download=True, transform=tx),
                       batch_size=batch_size, shuffle=shuffle, **kwargs)
    s2 = DataLoader(datasets.SVHN(path, split='test', download=True, transform=tx),
                      batch_size=batch_size, shuffle=shuffle, **kwargs)

    train_mnist_svhn = TensorDataset([
        ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
        ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
    ])
    test_mnist_svhn = TensorDataset([
        ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
        ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
    ])

    return train_mnist_svhn, test_mnist_svhn