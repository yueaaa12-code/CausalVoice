"""
CausalVoice Model: SynthesizerTrn with Causal Projector.

Extends O²-VC's SynthesizerTrn by adding a causal projection head
after the content encoder, enabling causal regularization during training.

The projector is only used during training for computing causal losses;
it does not affect inference.
"""

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from causal.projector import CausalProjector


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dilation_rate,
                n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate,
                              n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def forward_with_hidden(self, x, x_lengths, g=None):
        """Forward that also returns the hidden (pre-projection) representation."""
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        hidden = self.enc(x, x_mask, g=g)
        stats = self.proj(hidden) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask, hidden


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock_cls = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2**i),
                                upsample_initial_channel // (2**(i+1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3,
                 model_hidden_size=256, model_embedding_size=256):
        super().__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]
        if mel_len > partial_frames:
            mel_slices = []
            for i in range(0, mel_len - partial_frames, partial_hop):
                mel_slices.append(mel[:, i:i + partial_frames])
            mel_slices.append(last_mel)
            mels = torch.stack(mel_slices, 0).squeeze(1)
            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
        else:
            with torch.no_grad():
                embed = self(last_mel)
        return embed


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs += [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrnCausal(nn.Module):
    """
    CausalVoice Synthesizer: O²-VC SynthesizerTrn + Causal Projector.
    
    The causal projector is attached to the content encoder (enc_p) and
    produces embeddings for causal regularization during training.
    At inference time, the projector is not used.
    """

    def __init__(self, spec_channels, segment_size, inter_channels,
                 hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes,
                 gin_channels, ssl_dim, use_spk, **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.use_spk = use_spk

        # Content encoder (frozen, extracts content from WavLM features)
        self.enc_p = Encoder(ssl_dim, inter_channels, hidden_channels, 5, 1, 16)
        # Posterior encoder
        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                             gin_channels=gin_channels)
        # Decoder (HiFi-GAN generator)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes,
                             resblock_dilation_sizes, upsample_rates,
                             upsample_initial_channel, upsample_kernel_sizes,
                             gin_channels=gin_channels)
        # Normalizing flow
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4,
                                         gin_channels=gin_channels)

        # F0 conditioning
        self.f0_bins = torch.arange(2, 1024, 2)
        self.f0_emb = nn.Embedding(512, hidden_channels)
        self.f0_conv = nn.Conv1d(hidden_channels, hidden_channels, 1)

        # Speaker conditioning
        if self.use_spk:
            # Learnable speaker embedding table
            n_speakers = kwargs.get('n_speakers', 256)
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.enc_spk = SpeakerEncoder(model_hidden_size=gin_channels,
                                          model_embedding_size=gin_channels)

        # ===================== CAUSAL PROJECTOR (NEW) =====================
        # Projects content encoder hidden states to causal embedding space
        self.causal_projector = CausalProjector(
            hidden_channels=hidden_channels,
            temporal_pool='mean',
            proj_dim=32  # compact embedding for contrastive learning
        )

        # Freeze content extractor by default (same as O²-VC)
        # Can be unfrozen during causal training (controlled by config)
        self._enc_p_frozen = True
        for param in self.enc_p.parameters():
            param.requires_grad = False
    
    def unfreeze_enc_p(self):
        """Unfreeze content encoder for causal gradient flow."""
        if self._enc_p_frozen:
            for param in self.enc_p.parameters():
                param.requires_grad = True
            self._enc_p_frozen = False

    def forward(self, c, spec, g=None, f0=None, mel=None, c_lengths=None, spec_lengths=None):
        """Standard forward for VC training (same as O²-VC)."""
        if c_lengths is None:
            c_lengths = torch.LongTensor([c.size(-1)] * c.size(0)).to(c.device)
        if spec_lengths is None:
            spec_lengths = torch.LongTensor([spec.size(-1)] * spec.size(0)).to(spec.device)

        if self.use_spk:
            g = self.emb_g(g).unsqueeze(-1)    # (B,) → (B, gin_channels, 1)
        else:
            g = self.enc_spk(mel.transpose(1, 2))
            g = g.unsqueeze(-1)

        # F0 embedding
        f0 = torch.bucketize(f0, self.f0_bins.to(f0.device)).to(f0.device).squeeze(1)
        f0 = self.f0_emb(f0).transpose(1, 2)
        f0 = self.f0_conv(f0)

        # Content encoding
        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        # Flow
        z_p = self.flow(z, spec_mask, g=g)

        # Slice and decode
        z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
        f0_slice = commons.slice_segments(f0, ids_slice, self.segment_size)
        o = self.dec(z_slice + f0_slice, g=g)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def get_causal_embeddings(self, c, spk=None, c_lengths=None):
        """
        Extract causal embeddings from synthetic data for regularization.
        
        Args:
            c: (B, ssl_dim, T) WavLM features of synthetic utterances
            spk: unused (causal projector doesn't need speaker info)
            c_lengths: (B,) length of each sequence
            
        Returns:
            embeds: (B, proj_dim) L2-normalized causal embeddings
        """
        if c_lengths is None:
            c_lengths = torch.LongTensor([c.size(-1)] * c.size(0)).to(c.device)

        # Get hidden representation from content encoder
        _, _, _, x_mask, hidden = self.enc_p.forward_with_hidden(c, c_lengths)

        # Project to causal embedding space
        embeds = self.causal_projector(hidden, mask=x_mask)

        return embeds

    def infer(self, c, g=None, f0=None, mel=None, c_lengths=None):
        """Inference (same as O²-VC, projector not used)."""
        if c_lengths is None:
            c_lengths = torch.LongTensor([c.size(-1)] * c.size(0)).to(c.device)
        if self.use_spk:
            if g is not None and g.dim() == 1:
                # g is speaker ID tensor
                g = self.emb_g(g).unsqueeze(-1)
            elif g is not None and g.dim() == 2:
                # g is already an embedding (B, gin_channels)
                g = g.unsqueeze(-1)
        else:
            if mel is not None:
                g = self.enc_spk.embed_utterance(mel.transpose(1, 2))
                g = g.unsqueeze(-1)
            else:
                g = None

        f0 = torch.bucketize(f0, self.f0_bins.to(f0.device)).to(f0.device).squeeze(1)
        f0 = self.f0_emb(f0).transpose(1, 2)
        f0 = self.f0_conv(f0)

        z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        f0 = f0[:, :, :z.size(2)]
        o = self.dec((z + f0) * c_mask, g=g)
        return o
