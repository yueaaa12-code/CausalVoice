# CausalVoice: Causal Sim-to-Real Transfer for Voice Conversion

A causally-aware voice conversion framework that leverages synthetic data interventions to learn domain-invariant representations, bridging the sim-to-real gap in any-to-any voice conversion.

## Key Idea

In synthetic (TTS-generated) data, we can freely **intervene** on factors like speaker identity, prosody, and noise — something impossible with real recordings. We exploit this to:

1. **Generate causal labels**: measure how much each intervention changes the output
2. **Learn causal-invariant representations**: via contrastive/ranking regularization on the encoder
3. **Transfer to real data**: the shared encoder produces representations robust to synthetic artifacts

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CausalVoice Training                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Real Data Stream:                                        │
│    source_audio → WavLM → enc_p → flow → dec → wav_out   │
│    Loss: L_mel + L_kl + L_fm + L_gen (standard VC loss)   │
│                                                           │
│  Synthetic Data Stream (with interventions):              │
│    factual_audio   → WavLM → enc_p → projector → embed_f │
│    intervened_audio → WavLM → enc_p → projector → embed_c │
│    Loss: L_contrastive + L_ranking (causal regularization)│
│                                                           │
│  Total Loss = L_vc + λ_cont * L_contrastive               │
│                     + λ_rank * L_ranking                   │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
conda create -n causalvoice python==3.10.12
conda activate causalvoice
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data Preparation

### 1. Real Data (e.g., VCTK)
Standard voice conversion pairs: `filelists/train_real.txt`

### 2. Synthetic Data with Causal Interventions
Generate with a multi-speaker TTS, producing groups:
- Factual: original synthesis
- Counterfactual: same text, different speaker/prosody/noise

Run intervention data generation:
```bash
python causal/generate_interventions.py \
  --tts_model <path_to_multispeaker_tts> \
  --output_dir data/synthetic_causal/ \
  --num_samples 10000
```

### 3. Preprocessing
```bash
python preprocess_ssl.py   # Extract WavLM features
python preprocess_spk.py   # Extract speaker embeddings
python preprocess_f0.py    # Extract F0
```

## Training

### Baseline (standard O²-VC)
```bash
python train_f0.py -c configs/freevc_f0.json -m logs/baseline
```

### CausalVoice (with causal regularization)
```bash
python train_causal.py -c configs/causal_vc.json -m logs/causal_voice \
  --lambda_contrastive 1000.0 \
  --lambda_ranking 1000.0 \
  --causal_data_path data/synthetic_causal/
```

## Evaluation

```bash
# Standard VC metrics (WER, speaker similarity)
python evaluate.py --checkpoint logs/causal_voice/G_best.pth

# Causal awareness metrics (HNC, ARS)
python evaluate_causal.py --checkpoint logs/causal_voice/G_best.pth \
  --causal_test_path data/synthetic_causal/test/
```

## Citation

```bibtex
@article{causalvoice2025,
  title={CausalVoice: Causal Sim-to-Real Transfer for Voice Conversion},
  year={2025}
}
```

## Acknowledgements

Built upon:
- [O²-VC](https://github.com/huutuongtu/OOVC) (EMNLP 2025)
- [CausalSim2Real](https://github.com/vita-epfl/CausalSim2Real) (CVPR 2025)
- [FreeVC](https://github.com/OlaWod/FreeVC) / [VITS](https://github.com/jaywalnut310/vits)
