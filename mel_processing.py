"""
Mel spectrogram processing utilities for CausalVoice.
Adapted from VITS/FreeVC/O²-VC.
"""

import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


class MelScale(torch.nn.Module):
    """Pre-computed mel filterbank."""
    def __init__(self, n_fft, n_mels, sample_rate, f_min, f_max):
        super().__init__()
        mel_basis = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
                                   fmin=f_min, fmax=f_max)
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())

    def forward(self, spectrogram):
        return torch.matmul(self.mel_basis, spectrogram)


# Global cache for mel filterbanks and hann windows
mel_basis_cache = {}
hann_window_cache = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window_cache
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window_cache:
        hann_window_cache[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1),
                                (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y.float(), n_fft, hop_length=hop_size, win_length=win_size,
                      window=hann_window_cache[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False,
                      onesided=True, return_complex=True)
    spec = torch.abs(spec)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis_cache
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        mel_basis_cache[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis_cache[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax,
                          center=False):
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)
    spec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
    return spec
