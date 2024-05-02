import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from byol import BYOL
from ts2vec import TS2Vec
from ts2vec.utils import split_with_nan, centerize_vary_length_series, take_per_row
from ts2vec_byol.arch import SimpleArchitecture, LSTMArchitecture, ProjectorArchitecture, TS2VecProjectorArchitecture
from ts2vec_byol.losses import cross_cosine_loss, hierarchical_loss


class TS2VecBYOL(TS2Vec):
    class CropOverlap(nn.Module):

        def __init__(self, crop_l_mutable: list[int]):
            # mutable arguments are necessary to be able to update the value of crop_l later
            super().__init__()
            self.crop_l_mutable = crop_l_mutable
            self.n_aug = 0

        def forward(self, x):
            crop_l = self.crop_l_mutable[0]
            if self.n_aug % 2 == 0:
                x = x[:, -crop_l:]
            else:
                x = x[:, :crop_l]
            # strong assumption: each two calls are on two partially overlapping crops of the same series
            self.n_aug = (self.n_aug + 1)
            return x

    def __init__(self,
                 input_dims: int,
                 output_dims: int = 320,
                 hidden_dims: int = 64,
                 depth: int = 10,
                 device: int = 'cuda',
                 lr: int = 0.0003,
                 batch_size: int = 16,
                 max_train_length=None,
                 temporal_unit: int = 0,
                 after_iter_callback=None,
                 after_epoch_callback=None,
                 use_momentum=True,
                 hier_loss=False,
                 pred_hidden_dims=128,
                 arch=None,  # None, 'lstm', 'proj', 'proj2'
                 proj_hidden_dims=128,
                 proj_output_dims=64,
                 ):
        super().__init__(input_dims, output_dims, hidden_dims, depth, device, lr, batch_size, max_train_length,
                         temporal_unit, after_iter_callback, after_epoch_callback)

        self.arch = arch
        if arch is None:
            encoder, projector, predictor = SimpleArchitecture(
                input_dims, output_dims, hidden_dims, depth, pred_hidden_dims
            )
        elif arch == 'lstm':
            encoder, projector, predictor = LSTMArchitecture(
                input_dims, output_dims, hidden_dims, depth, proj_hidden_dims, proj_output_dims, pred_hidden_dims,
                hier_loss
            )
        elif arch == 'proj':
            encoder, projector, predictor = ProjectorArchitecture(
                input_dims, output_dims, hidden_dims, depth, proj_hidden_dims, proj_output_dims, pred_hidden_dims
            )
        elif arch == 'proj2':
            encoder, projector, predictor = TS2VecProjectorArchitecture(
                input_dims, output_dims, hidden_dims, depth, proj_hidden_dims, proj_output_dims, pred_hidden_dims
            )
        else:
            return  # raise exception
        encoder, predictor = encoder.to(device), predictor.to(device)
        if projector is not None:
            projector = projector.to(device)
        self.encoder, self.projector, self.predictor = encoder, projector, predictor

        if hier_loss:
            loss_fn = hierarchical_loss
        else:
            loss_fn = cross_cosine_loss

        self._net, self.net = None, encoder
        self._crop_l_mutable = [0]

        self._learner = BYOL(
            encoder=self._wrap_encoder(),
            predictor=predictor,
            augment_fn=self.augment_fn,
            loss_fn=loss_fn,
            device=self.device,
            use_momentum=use_momentum
        )
        self._learner.target_encoder[1].crop_l_mutable = self._crop_l_mutable

    def _wrap_encoder(self):
        if self.projector is None:
            encoder = nn.Sequential(
                self.encoder,
                self.CropOverlap(self._crop_l_mutable).to(self.device),
            ).to(self.device)
        else:
            encoder = nn.Sequential(
                self.encoder,
                self.CropOverlap(self._crop_l_mutable).to(self.device),
                self.projector
            ).to(self.device)
        return encoder

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        assert train_data.ndim == 3

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        if n_iters is None:
            n_iters = n_epochs * (len(train_data) // min(self.batch_size, len(train_data)))
        iters_bar = tqdm(total=n_iters, desc='Iterations', position=0, leave=True)

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True,
                                  drop_last=True)

        optimizer = torch.optim.AdamW(self._learner.parameters(), lr=self.lr)

        losses = []

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0  # count the iterations in each epoch

            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset: window_offset + self.max_train_length]
                x = x.to(self.device)

                optimizer.zero_grad()
                loss = self._learner(x)
                loss.backward()
                optimizer.step()
                if self._learner.use_momentum:
                    self._learner.update_moving_average()

                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1
                iters_bar.update(1)

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            losses.append(cum_loss)
            self.n_epochs += 1
            if verbose:
                print(f'Epoch {self.n_epochs}: loss={cum_loss:.4f}')

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        self.net = self._learner.online_encoder[0]  # only TSEncoder
        return losses

    def augment_fn(self, x):
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
        self._crop_l_mutable[0] = crop_l

        aug_one = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        aug_two = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)

        return aug_one, aug_two


class TS2VecBYOLAugs(TS2VecBYOL):

    def __init__(self,
                 input_dims: int,
                 output_dims: int = 320,
                 hidden_dims: int = 64,
                 depth: int = 10,
                 device: int = 'cuda',
                 lr: int = 0.0003,
                 batch_size: int = 16,
                 max_train_length=None,
                 temporal_unit: int = 0,
                 after_iter_callback=None,
                 after_epoch_callback=None,
                 use_momentum=True,
                 hier_loss=False,
                 pred_hidden_dims=128,
                 arch=None,  # None, 'lstm', 'proj'
                 proj_hidden_dims=128,
                 proj_output_dims=64,
                 ):
        super().__init__(input_dims, output_dims, hidden_dims, depth, device, lr, batch_size, max_train_length,
                         temporal_unit, after_iter_callback, after_epoch_callback, use_momentum, hier_loss,
                         pred_hidden_dims, arch, proj_hidden_dims, proj_output_dims)
        self.p, self.sigma = 0.5, 0.5
        if hier_loss:
            self._learner.loss_fn = hierarchical_loss
        else:
            self._learner.loss_fn = cross_cosine_loss

    def augment_fn(self, x):
        aug_one = self.jitter(self.shift(self.scale(x)))
        aug_two = self.jitter(self.shift(self.scale(x)))
        return aug_one, aug_two

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.shape).to(self.device) * self.sigma)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (torch.randn(x.size(-1)).to(self.device) * self.sigma + 1)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.size(-1)).to(self.device) * self.sigma)

    def _wrap_encoder(self):
        if self.projector is None:
            encoder = nn.Sequential(
                self.encoder,
            ).to(self.device)
        else:
            encoder = nn.Sequential(
                self.encoder,
                self.projector
            ).to(self.device)
        return encoder
