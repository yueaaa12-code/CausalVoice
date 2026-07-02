"""
Utility functions for CausalVoice (adapted from VITS/FreeVC/O²-VC).
"""

import torch
import numpy as np


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def sequence_mask(length, max_len=None):
    if max_len is None:
        max_len = length.max()
    x = torch.arange(max_len, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str_max = ids_str_max.clamp(min=1)
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        if idx_end > x.size(2):
            idx_end = x.size(2)
            idx_str = max(0, idx_end - segment_size)
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def clip_grad_value_(parameters, clip_value):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if clip_value is not None:
        clip_value = float(clip_value)
        for p in parameters:
            p.grad.data.clamp_(-clip_value, clip_value)
    # Return gradient norm for logging
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
