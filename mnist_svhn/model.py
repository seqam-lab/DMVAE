import torch
import torch.nn as nn

import sys

sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs
from util import kaiming_init

EPS = 1e-9
TEMP = 0.66


class EncoderA(nn.Module):
    def __init__(self, seed, num_pixels=784,
                 num_hidden=256,
                 zShared_dim=10,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.seed = seed

        self.enc_hidden = nn.Sequential(
            nn.Linear(num_pixels, num_hidden),
            nn.ReLU())

        self.fc = nn.Linear(num_hidden, 2 * zPrivate_dim + 2 * zShared_dim)
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    @expand_inputs
    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        hiddens = self.enc_hidden(x)
        stats = self.fc(hiddens)

        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)

        muShared = stats[:, :, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q.normal(loc=muPrivate,
                 scale=stdPrivate,
                 name='privateA')

        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedA')

        return q


class DecoderA(nn.Module):
    def __init__(self, seed, num_pixels=784,
                 num_hidden=256,
                 zShared_dim=10,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP

        self.style_mean = zPrivate_dim
        self.style_std = zPrivate_dim
        self.num_digits = zShared_dim
        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(zPrivate_dim + zShared_dim, num_hidden),
            nn.ReLU())
        self.dec_image = nn.Sequential(
            nn.Linear(num_hidden, num_pixels),
            nn.Sigmoid())
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, images, shared, q=None, p=None, num_samples=None):
        private_mean = torch.zeros_like(q['privateA'].dist.loc)
        private_std = torch.ones_like(q['privateA'].dist.scale)
        shared_mean = torch.zeros_like(q['sharedA'].dist.loc)
        shared_std = torch.ones_like(q['sharedA'].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(private_mean,
                            private_std,
                            value=q['privateA'],
                            name='privateA')
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.normal(shared_mean,
                               shared_std,
                               value=shared[shared_name],
                               name=shared_name)

            hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))

            images_mean = self.dec_image(hiddens).squeeze(0)


            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
                   images_mean, images, name='images1_' + shared_name)
        return p


    def forward2(self, z):
        hiddens = self.dec_hidden(z)
        images_mean = self.dec_image(hiddens)
        return images_mean


class EncoderB(nn.Module):
    def __init__(self, seed,
                 zShared_dim=10,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.seed = seed

        self.enc_hidden = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2 * zPrivate_dim + 2 * zShared_dim))
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        hiddens = self.enc_hidden(x)
        hiddens = hiddens.view(hiddens.size(0), -1)
        stats = self.fc(hiddens)
        stats = stats.unsqueeze(0)
        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)

        muShared = stats[:, :, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)


        q.normal(loc=muPrivate,
                 scale=stdPrivate,
                 name='privateB')

        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedB')

        return q


class DecoderB(nn.Module):
    def __init__(self, seed,
                 zShared_dim=10,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP

        self.num_digits = zShared_dim
        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(zPrivate_dim + zShared_dim, 256 * 2 * 2),
            nn.ReLU())
        self.dec_image = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid())
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, images, shared, q=None, p=None, num_samples=None):
        private_mean = torch.zeros_like(q['privateB'].dist.loc)
        private_std = torch.ones_like(q['privateB'].dist.scale)
        shared_mean = torch.zeros_like(q['sharedB'].dist.loc)
        shared_std = torch.ones_like(q['sharedB'].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(private_mean,
                            private_std,
                            value=q['privateB'],
                            name='privateB')
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.normal(shared_mean,
                               shared_std,
                               value=shared[shared_name],
                               name=shared_name)

            hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))
            hiddens = hiddens.view(-1, 256, 2, 2)
            images_mean = self.dec_image(hiddens)

            images_mean = images_mean.view(images_mean.size(0), -1)
            images = images.view(images.size(0), -1)

            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
                   images_mean, images, name='images2_' + shared_name)
        return p

    def forward2(self, z):
        hiddens = self.dec_hidden(z)
        hiddens = hiddens.view(-1, 256, 2, 2)
        images_mean = self.dec_image(hiddens)
        return images_mean
