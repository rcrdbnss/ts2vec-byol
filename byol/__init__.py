import copy
from functools import wraps

import torch
import torch.distributed as dist
from torch import nn


# helper functions

def default(val, def_val):
    return def_val if val is None else val


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def MaybeSyncBatchnorm(is_distributed=None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


# exponential moving average

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


class BYOL(nn.Module):
    def __init__(
            self,
            encoder,
            predictor,
            augment_fn,
            loss_fn,
            device,
            # mock_data,
            moving_average_decay=0.99,
            use_momentum=True,
    ):
        super().__init__()

        self.augment_fn = augment_fn
        self.loss_fn = loss_fn

        self.online_encoder = encoder

        self.use_momentum = use_momentum
        self.target_ema_updater = EMA(moving_average_decay)
        self.target_encoder = None
        self._get_target_encoder()

        self.online_predictor = predictor

        # get device of network and make wrapper same device
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        # self.forward(mock_data)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
            self,
            x,
            return_embedding=False,
            return_projection=True
    ):
        if self.training and x.shape[0] == 1:
            print('you must have greater than 1 sample when training, due to the batchnorm in the projection layer')

        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        aug_one, aug_two = self.augment_fn(x)

        online_proj_one, online_proj_two = self.online_encoder(aug_one.clone()), self.online_encoder(aug_two.clone())
        online_pred_one, online_pred_two = self.online_predictor(online_proj_one), self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder  # deepcopy of initial online encoder if BYOL

            target_proj_one, target_proj_two = target_encoder(aug_one), target_encoder(aug_two)
            target_proj_one, target_proj_two = target_proj_one.detach(), target_proj_two.detach()

        loss = self.loss_fn(online_pred_one, online_pred_two, target_proj_one, target_proj_two)
        return loss.mean()
        # return loss
