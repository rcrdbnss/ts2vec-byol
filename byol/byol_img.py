import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision import transforms as T

from byol import BYOL, MLP, singleton, MaybeSyncBatchnorm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.empty_cache()

resnet = models.resnet50(pretrained=True).to('cuda')

projection_size = 256
projection_hidden_size = 4096
image_size = 256


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False, sync_batchnorm = None):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, sync_batchnorm = self.sync_batchnorm)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True, return_representation=False):
        assert return_projection or return_representation, 'must return at least one between projection and representation'
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        if not return_representation:
            return projection
        return projection, representation


encoder = NetWrapper(
    resnet,
    projection_size,
    projection_hidden_size,
    layer='avgpool',
    use_simsiam_mlp=False,
    sync_batchnorm=None
)

predictor = MLP(projection_size, projection_size, projection_hidden_size)


def get_module_device(module):
    return next(module.parameters()).device


device = get_module_device(resnet)


# augmentation utils
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


DEFAULT_AUG = torch.nn.Sequential(
    RandomApply(
        T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        p=0.3
    ),
    T.RandomGrayscale(p=0.2),
    T.RandomHorizontalFlip(),
    RandomApply(
        T.GaussianBlur((3, 3), (1.0, 2.0)),
        p=0.2
    ),
    T.RandomResizedCrop((image_size, image_size)),
    T.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])),
).to(device)


def augment_fn(x):
    return DEFAULT_AUG(x), DEFAULT_AUG(x)


# loss fn
def loss_fn(online_pred_1, online_pred_2, target_proj_1, target_proj_2):
    def fn(x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    loss_one = fn(online_pred_1, target_proj_2.detach())
    loss_two = fn(online_pred_2, target_proj_1.detach())
    loss = loss_one + loss_two
    return loss


learner = BYOL(
    encoder=encoder,
    predictor=predictor,
    augment_fn=augment_fn,
    loss_fn=loss_fn,
    device=device,
    # mock_data=torch.randn(2, 3, image_size, image_size, device=device),
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)


for e in range(100):
    print(e, end=' ')
    images = sample_unlabelled_images().to(device)
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average()  # update moving average of target encoder

# save your improved network
# torch.save(resnet.state_dict(), './improved-net.pt')
def flatten(t):
    return t.reshape(t.shape[0], -1)


# SimSiam projector
def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )
