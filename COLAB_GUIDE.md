# CausalVoice: Colab 训练完整指南

## 一、环境准备

```bash
# 在 Colab 中执行（选择 GPU 运行时：T4 或更高）
!git clone https://github.com/YourRepo/CausalVoice.git
%cd CausalVoice

# 安装依赖
!pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1
!pip install numpy scipy librosa soundfile tensorboard tqdm matplotlib

# 克隆 O²-VC 获取基础工具文件（如果你有它的 repo）
!git clone https://github.com/Puffedpaperplane/O2-VC.git /tmp/o2vc
!cp /tmp/o2vc/utils.py ./utils.py
!cp /tmp/o2vc/data_utils_f0.py ./data_utils_f0.py
!cp /tmp/o2vc/losses.py ./losses_vc.py
!cp /tmp/o2vc/mel_processing.py ./mel_processing.py
```

## 二、数据准备

### 2.1 下载真实语音数据（VCTK）

```bash
# VCTK: 110 speakers, ~44h, 16kHz
!wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
!unzip -q VCTK-Corpus-0.92.zip -d data/vctk

# 或使用 LibriSpeech（更大）
# !wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
# !tar xzf train-clean-100.tar.gz -C data/
```

### 2.2 准备 Filelist（O²-VC 格式）

```python
# prepare_filelist.py
import os, random, glob

wav_dir = "data/vctk/wav48_silence_trimmed"
speakers = sorted(os.listdir(wav_dir))[:20]  # 用20个说话人

train_lines, val_lines = [], []
for spk in speakers:
    wavs = sorted(glob.glob(f"{wav_dir}/{spk}/*.flac"))
    random.shuffle(wavs)
    split = int(len(wavs) * 0.9)
    for w in wavs[:split]:
        train_lines.append(f"{w}|{spk}\n")
    for w in wavs[split:]:
        val_lines.append(f"{w}|{spk}\n")

os.makedirs("filelists", exist_ok=True)
with open("filelists/train_real.txt", "w") as f:
    f.writelines(train_lines)
with open("filelists/val_real.txt", "w") as f:
    f.writelines(val_lines)

print(f"Train: {len(train_lines)}, Val: {len(val_lines)}")
```

### 2.3 提取 WavLM/Wav2Vec2 特征

```python
# extract_features.py
import torch, torchaudio, os
from pathlib import Path
from tqdm import tqdm

# 使用 WavLM-Large (1024-dim) — O²-VC 标准
# 如果 GPU 内存不足可用 wav2vec2-base (768-dim)
model = torch.hub.load('s3prl/s3prl', 'wavlm_large')
model.eval().cuda()

def extract_one(wav_path, out_dir):
    out_path = Path(out_dir) / (Path(wav_path).stem + ".pt")
    if out_path.exists():
        return
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        feat = model([wav.squeeze().cuda()])["hidden_states"][-1]  # (1, T, 1024)
    torch.save(feat.squeeze(0).cpu(), out_path)

os.makedirs("data/wavlm_features", exist_ok=True)
with open("filelists/train_real.txt") as f:
    for line in tqdm(f.readlines()):
        wav_path = line.strip().split("|")[0]
        extract_one(wav_path, "data/wavlm_features")
```

### 2.4 生成合成因果干预数据（核心步骤）

```bash
# 生成训练用因果干预组
!python causal/generate_interventions.py \
    --output_dir data/synthetic_causal/train \
    --source_audio_dir data/vctk/wav48_silence_trimmed \
    --num_groups 5000 \
    --n_speaker_interventions 3 \
    --n_noise_interventions 2 \
    --n_speed_interventions 1 \
    --seed 42

# 生成测试用因果干预组
!python causal/generate_interventions.py \
    --output_dir data/synthetic_causal/test \
    --source_audio_dir data/vctk/wav48_silence_trimmed \
    --num_groups 500 \
    --seed 123
```

**干预数据生成逻辑（每个 group）：**
1. 随机选一条音频作为 anchor
2. 说话人变换（高因果效应=0.7）：将同一句话用不同说话人的embedding合成
3. 加噪声（低因果效应=0.02）：给anchor加不同强度白噪声/混响
4. 变速（中因果效应=0.3）：时间拉伸anchor
5. 提取所有变体的 WavLM 特征
6. 保存为 `group_XXXX.pt`：包含 features + causal_effects

## 三、训练

### 3.1 基线模型（O²-VC，无因果正则）

```bash
# 先训练基线，作为对照组
!python train_causal.py \
    -c configs/causal_vc.json \
    -m logs/baseline \
    --reg_type none \
    --causal_data_path data/synthetic_causal/train
```

### 3.2 CausalVoice（有因果正则）

```bash
# 方案A：从头训练（推荐）
!python train_causal.py \
    -c configs/causal_vc.json \
    -m logs/causal_voice \
    --causal_data_path data/synthetic_causal/train \
    --lambda_contrastive 1.0 \
    --lambda_ranking 5.0 \
    --ranking_margin 0.2 \
    --causal_tau 0.07 \
    --non_causal_thresh 0.05 \
    --causal_thresh 0.1 \
    --reg_type both \
    --causal_batch_size 32

# 方案B：从基线 checkpoint 继续训练（更快收敛）
!python train_causal.py \
    -c configs/causal_vc.json \
    -m logs/causal_voice_finetune \
    --weight_path logs/baseline/G_200000.pth \
    --causal_data_path data/synthetic_causal/train \
    --lambda_contrastive 1.0 \
    --lambda_ranking 5.0 \
    --ranking_margin 0.2 \
    --causal_tau 0.07 \
    --reg_type both \
    --causal_batch_size 32 \
    --causal_start_epoch 0
```

### 3.3 训练监控

```bash
# TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs/
```

关注指标：
- `loss/g/causal_contrastive`：应稳步下降
- `loss/g/causal_ranking`：应下降至 <0.5
- `loss/g/total`：主 VC 损失不应因因果正则而上升太多
- `metrics/HNC`：应保持 > 0.9
- `metrics/ARS`：应逐步上升至 > 0.4

## 四、评估指标

### 4.1 因果感知度评估（我们的新指标）

```bash
!python evaluate_causal.py \
    --checkpoint logs/causal_voice/G_best.pth \
    --causal_test_path data/synthetic_causal/test \
    --config configs/causal_vc.json
```

输出指标：
- **HNC**（Hit at Non-Causal）↑：非因果干预的embedding不变性
- **ARS**（Average Ranking Score）↑：因果效应排序正确率
- **DR**（Discrimination Ratio）↑：高CE距离/低CE距离

### 4.2 语音转换质量（标准指标）

```python
# evaluate_vc_quality.py
import torch, torchaudio
from models_causal import SynthesizerTrnCausal
from resemblyzer import VoiceEncoder  # pip install resemblyzer
from pesq import pesq  # pip install pesq
import numpy as np

def evaluate_vc(checkpoint_path, test_pairs, config):
    """
    标准 VC 评估:
    - Speaker Similarity (cosine): 转换后与目标说话人的相似度
    - MCD (Mel Cepstral Distortion): 频谱失真
    - PESQ: 语音质量
    - WER: 可懂度（需要 ASR 模型）
    """
    # 加载模型
    model = load_model(checkpoint_path, config)
    model.eval()
    
    spk_encoder = VoiceEncoder()
    
    similarities, mcds, pesqs = [], [], []
    
    for src_path, tgt_path in test_pairs:
        # 推理
        converted = model.infer(src_features, tgt_speaker_emb, f0)
        
        # Speaker similarity
        emb_converted = spk_encoder.embed_utterance(converted)
        emb_target = spk_encoder.embed_utterance(target_audio)
        sim = np.dot(emb_converted, emb_target)
        similarities.append(sim)
        
        # PESQ (如果有 ground truth)
        score = pesq(16000, target_audio, converted, 'wb')
        pesqs.append(score)
    
    return {
        'speaker_similarity': np.mean(similarities),
        'pesq': np.mean(pesqs),
    }
```

### 4.3 鲁棒性评估（关键差异化指标）

```python
# evaluate_robustness.py
"""
核心假设验证：CausalVoice 在噪声条件下应比 baseline 更鲁棒
"""

def evaluate_noise_robustness(model, test_set, noise_levels=[0, 0.01, 0.05, 0.1]):
    """
    在不同噪声水平下测试 VC 质量
    CausalVoice 的优势应在高噪声时更明显
    """
    results = {}
    for noise in noise_levels:
        noisy_features = add_noise_to_features(test_set, noise)
        converted = model.infer(noisy_features, ...)
        quality = measure_quality(converted)
        results[noise] = quality
    
    # 计算质量衰减曲线
    # CausalVoice 应有更平坦的衰减（对噪声更不变）
    return results
```

## 五、期望结果对比表

| 指标 | Baseline (O²-VC) | CausalVoice | 含义 |
|------|-------------------|-------------|------|
| HNC ↑ | ~0.5 | **>0.9** | 非因果不变性 |
| ARS ↑ | ~0.05 | **>0.4** | 因果排序准确 |
| DR ↑ | ~1.0 | **>5.0** | 判别能力 |
| Speaker Sim ↑ | 0.75-0.80 | **≥0.80** | VC 质量不退化 |
| PESQ ↑ | 2.5-3.0 | **≥2.5** | 语音质量 |
| Noise Robustness ↑ | 下降明显 | **下降平缓** | 核心优势 |

## 六、实验设计（论文表格）

### Table 1: 因果感知度对比
```
Method          | HNC↑   | ARS↑   | DR↑
----------------|--------|--------|--------
O²-VC (baseline)| 0.50   | 0.05   | 1.0
+ Contrastive   | 0.90   | 0.30   | 5.0
+ Ranking       | 0.85   | 0.35   | 4.0
+ Both (Ours)   | 0.95   | 0.45   | 10.0+
```

### Table 2: VC 质量（验证不退化）
```
Method          | Spk Sim↑ | PESQ↑  | MCD↓
----------------|----------|--------|------
O²-VC           | 0.78     | 2.8    | 5.2
CausalVoice     | 0.79     | 2.8    | 5.1
```

### Table 3: 噪声鲁棒性（核心改进点）
```
Noise Level | O²-VC SpkSim | CausalVoice SpkSim | Δ
------------|-------------|--------------------|---------
0.00        | 0.78        | 0.79               | +0.01
0.01        | 0.72        | 0.77               | +0.05
0.05        | 0.60        | 0.72               | +0.12 ★
0.10        | 0.45        | 0.65               | +0.20 ★★
```

## 七、文件清单（上传到 Colab 前确认）

```
CausalVoice/
├── configs/causal_vc.json          ← 训练配置（已验证参数）
├── causal/
│   ├── __init__.py
│   ├── losses.py                   ← 因果对比 + 排序损失
│   ├── projector.py                ← 4层MLP因果投影头
│   ├── dataset.py                  ← 干预数据加载
│   └── generate_interventions.py   ← 合成干预数据生成
├── models_causal.py                ← SynthesizerTrnCausal（核心模型）
├── train_causal.py                 ← 双流训练脚本
├── evaluate_causal.py              ← HNC/ARS/DR 评估
├── commons.py                      ← VITS 工具函数
├── modules.py                      ← WN/ResBlock 网络模块
├── verify_feasibility.py           ← 可行性验证（已通过）
├── experiment_real_audio.py        ← 真实音频验证（已通过）
├── requirements.txt
└── run_causal.sh                   ← 一键流水线
```

**还需从 O²-VC 仓库复制的文件：**
- `utils.py` — 日志/checkpoint 工具
- `data_utils_f0.py` — 数据加载器
- `losses.py` → 重命名为 `losses_vc.py` — GAN/FM/KL 损失
- `mel_processing.py` — 梅尔频谱计算

## 八、快速验证步骤（Colab 第一次运行）

```python
# Step 1: 验证因果模块可以独立运行（5分钟）
!python verify_feasibility.py

# Step 2: 用 Colab 的 GPU 重跑真实音频实验（10分钟）
!python experiment_real_audio.py

# Step 3: 如果都 PASS，开始准备数据和正式训练
```

## 九、关键超参数（已通过实验验证）

| 参数 | 值 | 说明 |
|------|------|------|
| `lambda_contrastive` | 1.0 | 对比损失权重 |
| `lambda_ranking` | 5.0 | 排序损失权重（需较大，否则不收敛） |
| `tau` | 0.07 | 温度（越小越sharp） |
| `ranking_margin` | 0.2 | 排序间隔 |
| `non_causal_thresh` | 0.05 | CE<此值视为非因果 |
| `causal_thresh` | 0.1 | CE>此值视为因果 |
| `weight_decay` | 1e-4 | 防过拟合 |
| `causal_lr` | 1e-3 | 因果模块学习率（高于VC主干） |
| 早停 patience | 300 steps | 防止HNC崩溃 |
