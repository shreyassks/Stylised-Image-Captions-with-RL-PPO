import torch.nn as nn

from app.core.config import cfg


def activation(act):
    if act == 'RELU':
        return nn.ReLU(inplace=True)
    elif act == 'TANH':
        return nn.Tanh()
    elif act == 'GLU':
        return nn.GLU()
    elif act == 'ELU':
        return nn.ELU(cfg.MODEL.BILINEAR.ELU_ALPHA, inplace=True)
    elif act == 'CELU':
        return nn.CELU(cfg.MODEL.BILINEAR.ELU_ALPHA, inplace=True)
    elif act == 'GELU':
        return nn.GELU()
    else:
        return nn.Identity()


def expand_tensor(tensor, size, dim=1):
    if size == 1 or tensor is None:
        return tensor
    tensor = tensor.unsqueeze(dim)
    tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim + 1:])).contiguous()
    tensor = tensor.view(list(tensor.shape[:dim - 1]) + [-1] + list(tensor.shape[dim + 1:]))
    return tensor
