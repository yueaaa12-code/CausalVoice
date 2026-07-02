"""
CausalVoice Inference: Voice Conversion.

Usage:
    python convert.py --checkpoint logs/causal_voice/G_best.pth \
                      --source audio/source.wav \
                      --target audio/target.wav \
                      --output output/converted.wav \
                      --config configs/causal_vc.json
"""

import os
import argparse
import torch
import torchaudio
import numpy as np

from models_causal import SynthesizerTrnCausal
from mel_processing import mel_spectrogram_torch
import utils


def extract_f0(wav, sr=16000):
    """Extract F0 using torchaudio (FCPE/DIO alternative)."""
    # Simple pitch detection using autocorrelation
    # In production, use CREPE, FCPE, or DIO
    try:
        pitch = torchaudio.functional.detect_pitch_frequency(wav, sr, freq_low=50, freq_high=600)
        return pitch
    except:
        return torch.zeros(1, wav.size(-1) // 320)


def load_model(checkpoint_path, config_path, device='cuda'):
    """Load trained CausalVoice model."""
    hps = utils.get_hparams_from_file(config_path)
    model_cfg = hps.model if isinstance(hps.model, dict) else hps.model.__dict__
    data_cfg = hps.data if isinstance(hps.data, dict) else hps.data.__dict__
    
    n_fft = data_cfg.get('filter_length', 1280)
    
    model = SynthesizerTrnCausal(
        n_fft // 2 + 1,
        hps.train.segment_size // data_cfg.get('hop_length', 320),
        **model_cfg
    ).to(device)
    
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model.eval()
    return model, hps


def extract_ssl_features(wav_path, device='cuda', model_name='wavlm'):
    """
    Extract SSL (WavLM/wav2vec2) features from audio file.
    
    Falls back to wav2vec2-base if WavLM not available.
    """
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    
    # Try WavLM first, fallback to wav2vec2
    try:
        if model_name == 'wavlm':
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M  # or use huggingface WavLM
        else:
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
        ssl_model = bundle.get_model().to(device).eval()
        
        with torch.no_grad():
            features, _ = ssl_model.extract_features(wav.to(device))
            feat = features[-1].squeeze(0)  # (T, dim)
    except:
        # Fallback: use torchaudio wav2vec2
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        ssl_model = bundle.get_model().to(device).eval()
        with torch.no_grad():
            features, _ = ssl_model.extract_features(wav.to(device))
            feat = features[-1].squeeze(0)
    
    return feat.T  # (dim, T)


def get_speaker_embedding(wav_path, model, hps, device='cuda'):
    """Extract speaker embedding from target audio."""
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    
    data_cfg = hps.data if isinstance(hps.data, dict) else hps.data.__dict__
    n_mel = data_cfg.get('n_mel_channels', 80)
    
    # Compute mel for speaker encoder
    mel = mel_spectrogram_torch(
        wav.to(device), 
        data_cfg.get('filter_length', 1280),
        n_mel, 
        data_cfg.get('sampling_rate', 16000),
        data_cfg.get('hop_length', 320),
        data_cfg.get('win_length', 1280),
        data_cfg.get('mel_fmin', 0.0),
        data_cfg.get('mel_fmax', None)
    )
    
    # Use model's built-in speaker encoder
    if hasattr(model, 'enc_spk'):
        with torch.no_grad():
            spk_emb = model.enc_spk.embed_utterance(mel)
        return spk_emb
    else:
        # External speaker embedding (e.g., from speaker_encoder/)
        return None


@torch.no_grad()
def convert(model, source_features, target_spk_emb, f0, device='cuda', mel=None):
    """
    Run voice conversion inference.
    
    Args:
        model: trained SynthesizerTrnCausal
        source_features: (1, ssl_dim, T) content features from source
        target_spk_emb: (1, spk_dim) speaker embedding from target (or None if use_spk=False)
        f0: (1, 1, T) F0 contour
        mel: (1, n_mel, T_mel) mel spectrogram for speaker encoder (required if use_spk=False)
    
    Returns:
        audio: (1, 1, samples) converted waveform
    """
    c = source_features.to(device)
    c_lengths = torch.tensor([c.size(-1)], device=device)
    
    audio = model.infer(c, g=target_spk_emb, f0=f0, mel=mel, c_lengths=c_lengths)
    return audio


def main():
    parser = argparse.ArgumentParser(description="CausalVoice Conversion")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/causal_vc.json')
    parser.add_argument('--source', type=str, required=True, help='Source audio path')
    parser.add_argument('--target', type=str, required=True, help='Target speaker audio path')
    parser.add_argument('--output', type=str, default='output/converted.wav')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ssl_model', type=str, default='wav2vec2', choices=['wavlm', 'wav2vec2'])
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, hps = load_model(args.checkpoint, args.config, device)
    
    # Extract source content features
    print(f"Extracting features from source: {args.source}")
    source_feat = extract_ssl_features(args.source, device, args.ssl_model)
    source_feat = source_feat.unsqueeze(0)  # (1, dim, T)
    
    # Extract target speaker embedding
    print(f"Extracting speaker embedding from target: {args.target}")
    target_mel_wav, sr = torchaudio.load(args.target)
    if sr != 16000:
        target_mel_wav = torchaudio.functional.resample(target_mel_wav, sr, 16000)
    
    data_cfg = hps.data if isinstance(hps.data, dict) else hps.data.__dict__
    target_mel = mel_spectrogram_torch(
        target_mel_wav.to(device),
        data_cfg.get('filter_length', 1280),
        data_cfg.get('n_mel_channels', 80),
        data_cfg.get('sampling_rate', 16000),
        data_cfg.get('hop_length', 320),
        data_cfg.get('win_length', 1280),
        data_cfg.get('mel_fmin', 0.0),
        data_cfg.get('mel_fmax', None)
    )
    
    # Extract F0 from source
    source_wav, _ = torchaudio.load(args.source)
    f0 = extract_f0(source_wav).to(device)
    # Align F0 length with features
    T = source_feat.size(-1)
    if f0.size(-1) > T:
        f0 = f0[:, :T]
    else:
        f0 = torch.nn.functional.pad(f0, (0, T - f0.size(-1)))
    f0 = f0.unsqueeze(0)  # (1, 1, T)
    
    # Convert
    print("Converting...")
    spk_emb = get_speaker_embedding(args.target, model, hps, device)
    audio = convert(model, source_feat, spk_emb, f0, device, mel=target_mel)
    
    # Save output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    audio = audio.squeeze().cpu()
    # Normalize
    audio = audio / audio.abs().max() * 0.95
    torchaudio.save(args.output, audio.unsqueeze(0), 16000)
    print(f"Saved converted audio to {args.output}")


if __name__ == '__main__':
    main()
