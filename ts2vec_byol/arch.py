from torch import nn as nn

from ts2vec import TSEncoder
# from ts2vec_byol.losses import hierarchical_loss, cross_cosine_loss


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class ProjectorLSTM(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, output_dim_1=256):
        super().__init__()
        self.output_dim_1 = output_dim_1
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, output_dim)
        # )
        # self.lstm = nn.LSTM(output_dim, output_dim, batch_first=True)
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)

    def forward(self, x):
        # x = self.mlp(x)
        x, _ = self.lstm(x)
        _, T, _ = x.size()
        return x.flip(dims=(1,))[:, ::max(1, T // self.output_dim_1)].flip(dims=(1,))


def Predictor(input_dims, output_dims, hidden_dims):
    return nn.Sequential(
        nn.Linear(input_dims, hidden_dims),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dims, output_dims)
    )


def SimpleArchitecture(input_dims, output_dims, hidden_dims, depth, pred_hidden_dims):
    encoder = TSEncoder(input_dims, output_dims, hidden_dims, depth)
    predictor = Predictor(output_dims, output_dims, pred_hidden_dims)
    return encoder, None, predictor, None


def LSTMArchitecture(enc_input_dim, enc_output_dim, enc_hidden_dim, enc_depth,
                     proj_hidden_dim, proj_output_dim, pred_hidden_dim, hier_loss):
    encoder = TSEncoder(enc_input_dim, enc_output_dim, enc_hidden_dim, enc_depth)
    projector = ProjectorLSTM(enc_output_dim, proj_output_dim, proj_hidden_dim)
    predictor = Predictor(proj_output_dim, proj_output_dim, pred_hidden_dim)
    # loss = hierarchical_loss if hier_loss else cross_cosine_loss
    loss = None
    return encoder, projector, predictor, loss


def ProjectorArchitecture(enc_input_dim, enc_output_dim, enc_hidden_dim, enc_depth,
                          proj_hidden_dim, proj_output_dim, pred_hidden_dim):
    encoder = TSEncoder(enc_input_dim, enc_output_dim, enc_hidden_dim, enc_depth)
    projector = nn.Sequential(
        nn.Linear(enc_output_dim, proj_hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(proj_hidden_dim, proj_output_dim)
    )
    predictor = Predictor(proj_output_dim, proj_output_dim, pred_hidden_dim)
    return encoder, projector, predictor, None
