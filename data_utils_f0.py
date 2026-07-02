"""
Data loading utilities for CausalVoice.
Handles O²-VC format filelists and audio processing.

Filelist format: wav_path|speaker_id
Features are pre-extracted as .pt files (WavLM/wav2vec2).
"""

import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
from mel_processing import spectrogram_torch, mel_spectrogram_torch


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    Loads (content_features, spectrogram, waveform, speaker, f0) tuples.
    
    Expects:
    - Pre-extracted SSL features in feature_dir/{basename}.pt
    - Pre-extracted F0 in f0_dir/{basename}.pt  
    - Raw wav for spectrogram computation
    """

    def __init__(self, filelist_path, hparams):
        self.hparams = hparams
        self.filelist = self._load_filelist(filelist_path)
        
        # Data config
        h = hparams if isinstance(hparams, dict) else hparams.__dict__
        data = h.get('data', h)
        if hasattr(data, '__dict__'):
            data = data.__dict__
            
        self.sampling_rate = data.get('sampling_rate', 16000)
        self.filter_length = data.get('filter_length', 1280)
        self.hop_length = data.get('hop_length', 320)
        self.win_length = data.get('win_length', 1280)
        self.max_wav_value = data.get('max_wav_value', 32768.0)
        
        # Directories for pre-extracted features
        self.feature_dir = data.get('feature_dir', 'data/wavlm_features')
        self.f0_dir = data.get('f0_dir', 'data/f0')
        
        # Speaker to ID mapping
        self.speakers = sorted(list(set([item[1] for item in self.filelist])))
        self.spk2id = {spk: i for i, spk in enumerate(self.speakers)}
        
        # Train config
        train = h.get('train', h)
        if hasattr(train, '__dict__'):
            train = train.__dict__
        self.max_speclen = train.get('max_speclen', 128)
        
        random.seed(1234)
        random.shuffle(self.filelist)

    def _load_filelist(self, path):
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) >= 2:
                    items.append((parts[0], parts[1]))
                else:
                    items.append((parts[0], 'default'))
        return items

    def __getitem__(self, index):
        wav_path, spk = self.filelist[index]
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        
        # Load SSL features (content)
        feat_path = os.path.join(self.feature_dir, f"{basename}.pt")
        if os.path.exists(feat_path):
            c = torch.load(feat_path, weights_only=True)  # (T, dim)
            if c.dim() == 2:
                c = c.T  # → (dim, T)
        else:
            # Fallback: compute spectrogram as content (for testing without SSL)
            c = self._load_spec(wav_path)
        
        # Load waveform for spectrogram
        spec, wav = self._load_audio(wav_path)
        
        # Load F0
        f0_path = os.path.join(self.f0_dir, f"{basename}.pt")
        if os.path.exists(f0_path):
            f0 = torch.load(f0_path, weights_only=True)
        else:
            f0 = torch.zeros(1, spec.size(1))
        
        # Truncate to max_speclen
        spec_len = spec.size(1)
        if spec_len > self.max_speclen:
            start = random.randint(0, spec_len - self.max_speclen)
            spec = spec[:, start:start + self.max_speclen]
            wav = wav[:, start * self.hop_length:(start + self.max_speclen) * self.hop_length]
            # Align content features
            ssl_ratio = c.size(1) / spec_len if c.size(1) != spec_len else 1.0
            c_start = int(start * ssl_ratio)
            c_len = int(self.max_speclen * ssl_ratio)
            c = c[:, c_start:c_start + c_len]
            # Align F0
            if f0.dim() == 2:
                f0 = f0[:, start:start + self.max_speclen]
            else:
                f0 = f0[start:start + self.max_speclen].unsqueeze(0)
        
        # Speaker embedding (one-hot ID for now)
        spk_id = torch.LongTensor([self.spk2id[spk]])
        
        return c, spec, wav, spk_id, f0, f0

    def _load_audio(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav / self.max_wav_value if wav.abs().max() > 1.0 else wav
        
        # Compute spectrogram
        spec = spectrogram_torch(wav, self.filter_length, self.sampling_rate,
                                 self.hop_length, self.win_length, center=False)
        spec = spec.squeeze(0)  # (freq, T)
        return spec, wav

    def _load_spec(self, wav_path):
        """Fallback: use spectrogram as content features."""
        spec, _ = self._load_audio(wav_path)
        return spec

    def __len__(self):
        return len(self.filelist)


class TextAudioSpeakerCollate:
    """Collate function for variable-length sequences."""
    
    def __init__(self, hparams=None):
        self.hparams = hparams

    def __call__(self, batch):
        # batch: list of (c, spec, wav, spk_id, f0_src, f0_tgt)
        
        # Sort by spec length (descending)
        batch = sorted(batch, key=lambda x: x[1].size(1), reverse=True)
        
        max_spec_len = batch[0][1].size(1)
        max_wav_len = batch[0][2].size(1)
        max_c_len = max(item[0].size(1) for item in batch)
        
        c_dim = batch[0][0].size(0)
        spec_dim = batch[0][1].size(0)
        
        c_padded = torch.zeros(len(batch), c_dim, max_c_len)
        spec_padded = torch.zeros(len(batch), spec_dim, max_spec_len)
        wav_padded = torch.zeros(len(batch), 1, max_wav_len)
        spk_ids = torch.zeros(len(batch), dtype=torch.long)
        f0_src_padded = torch.zeros(len(batch), 1, max_spec_len)
        f0_tgt_padded = torch.zeros(len(batch), 1, max_spec_len)
        
        for i, (c, spec, wav, spk_id, f0_src, f0_tgt) in enumerate(batch):
            c_padded[i, :, :c.size(1)] = c
            spec_padded[i, :, :spec.size(1)] = spec
            wav_padded[i, :, :wav.size(1)] = wav
            spk_ids[i] = spk_id[0]
            if f0_src.dim() == 2:
                f0_src_padded[i, :, :f0_src.size(1)] = f0_src
                f0_tgt_padded[i, :, :f0_tgt.size(1)] = f0_tgt
            else:
                f0_src_padded[i, 0, :f0_src.size(0)] = f0_src
                f0_tgt_padded[i, 0, :f0_tgt.size(0)] = f0_tgt
        
        return c_padded, spec_padded, wav_padded, spk_ids, f0_src_padded, f0_tgt_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """Bucket sampler for efficient batching of variable-length sequences."""
    
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.epoch = 0
        
        # Use uniform bucketing instead of probing dataset
        n_items = len(dataset)
        self.batches = []
        indices = list(range(n_items))
        random.shuffle(indices)
        for i in range(0, n_items - batch_size + 1, batch_size):
            self.batches.append(indices[i:i + batch_size])

    def __iter__(self):
        random.seed(self.epoch)
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch
