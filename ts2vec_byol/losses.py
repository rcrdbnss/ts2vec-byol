import torch
from torch.nn import functional as F


def cos_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)



def cross_cosine_loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2):
    loss_one = cos_loss(online_pred_1, target_proj_2.detach())
    loss_two = cos_loss(online_pred_2, target_proj_1.detach())
    loss = loss_one + loss_two
    return loss.mean(dim=-1)


def hierarchical_loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2, loss_fn=cross_cosine_loss):
    loss, n = torch.zeros(online_pred_1.size(0), device=online_pred_1.device), 0
    while online_pred_1.size(1) > 1:
        loss += loss_fn(online_pred_1, online_pred_2, target_proj_1, target_proj_2)
        n += 1
        online_pred_1 = F.max_pool1d(online_pred_1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        online_pred_2 = F.max_pool1d(online_pred_2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        target_proj_1 = F.max_pool1d(target_proj_1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        target_proj_2 = F.max_pool1d(target_proj_2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    return loss / n
