import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np

def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)


def apply_poe(use_cuda, mu_sharedA, std_sharedA, mu_sharedB=None, std_sharedB=None):
    '''
    induce zS = encAB(xA,xB) via POE, that is,
        q(zI,zT,zS|xI,xT) := qI(zI|xI) * qT(zT|xT) * q(zS|xI,xT)
            where q(zS|xI,xT) \propto p(zS) * qI(zS|xI) * qT(zS|xT)
    '''
    EPS = 1e-9
    ZERO = torch.zeros(std_sharedA.shape)
    if use_cuda:
        ZERO = ZERO.cuda()

    logvar_sharedA = torch.log(std_sharedA ** 2 + EPS)

    if mu_sharedB is not None and std_sharedB is not None:
        logvar_sharedB = torch.log(std_sharedB ** 2 + EPS)
        logvarS = -logsumexp(
            torch.stack((ZERO, -logvar_sharedA, -logvar_sharedB), dim=2),
            dim=2
        )
    else:
        logvarS = -logsumexp(
            torch.stack((ZERO, -logvar_sharedA), dim=2),
            dim=2
        )
    stdS = torch.sqrt(torch.exp(logvarS))

    if mu_sharedB is not None and std_sharedB is not None:
        muS = (mu_sharedA / (std_sharedA ** 2 + EPS) +
               mu_sharedB / (std_sharedB ** 2 + EPS)) * (stdS ** 2)
    else:
        muS = (mu_sharedA / (std_sharedA ** 2 + EPS)) * (stdS ** 2)

    return muS, stdS



def kaiming_init(m, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


###  from https://github.com/iffsid/mmvae

def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)

###  from https://github.com/iffsid/mmvae
def unpack_data(dataB, device='cuda'):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if is_multidata(dataB):
        if torch.is_tensor(dataB[0]):
            if torch.is_tensor(dataB[1]):
                return dataB[0].to(device)  # mnist, svhn, cubI
            elif is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
            else:
                raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[1])))

        elif is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-svhn, cubIS
        else:
            raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[0])))
    elif torch.is_tensor(dataB):
        return dataB.to(device)
    else:
        raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB)))

