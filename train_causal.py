"""
CausalVoice Training Script.

Dual-stream training combining:
1. Real data stream: standard VC loss (mel + KL + GAN + feature matching)
2. Synthetic causal stream: contrastive + ranking regularization

Based on O²-VC (EMNLP 2025) training loop with CausalSim2Real (CVPR 2025) 
causal regularization framework.
"""

import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import commons
import utils
from data_utils_f0 import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler
)
from models_causal import SynthesizerTrnCausal, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from causal import (
    calc_contrastive_loss,
    calc_ranking_loss,
    calc_causal_awareness_metrics,
    CausalInterventionDataset,
    causal_collate_fn
)

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    assert torch.cuda.is_available(), "CUDA required for training."
    
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()

    if n_gpus > 1:
        # Multi-GPU: use DDP with mp.spawn
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(hps.train.port) if hasattr(hps.train, 'port') else '6667'
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    else:
        # Single-GPU (Colab): run directly without DDP
        run(0, 1, hps)


def run(rank, n_gpus, hps):
    global global_step
    use_ddp = (n_gpus > 1)

    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    if use_ddp:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    # ======================== DATA LOADERS ========================
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    train_sampler = DistributedBucketSampler(
        train_dataset, hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus, rank=rank, shuffle=True)
    collate_fn = TextAudioSpeakerCollate(hps)
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False,
                              pin_memory=True, collate_fn=collate_fn,
                              batch_sampler=train_sampler)

    # Causal intervention data loader
    causal_dataset = CausalInterventionDataset(hps.causal.data_path, hps)
    causal_loader = DataLoader(
        causal_dataset, batch_size=hps.causal.batch_size,
        shuffle=True, num_workers=2,
        collate_fn=causal_collate_fn, drop_last=True)

    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
    eval_loader = DataLoader(eval_dataset, num_workers=4, shuffle=True,
                             batch_size=hps.train.batch_size,
                             pin_memory=False, drop_last=False, collate_fn=collate_fn)

    # ======================== MODELS ========================
    net_g = SynthesizerTrnCausal(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model.__dict__
    ).cuda(rank)

    net_d = MultiPeriodDiscriminator(
        hps.model.use_spectral_norm if hasattr(hps.model, 'use_spectral_norm') else False
    ).cuda(rank)

    # Unfreeze enc_p if causal regularization is active
    if hasattr(hps.causal, 'unfreeze_enc_p') and hps.causal.unfreeze_enc_p:
        if hps.causal.reg_type != 'none':
            net_g.unfreeze_enc_p()
            logger.info("Unfroze enc_p for causal gradient flow")

    # Optimizers (must be after unfreeze so enc_p params are included)
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate,
                                 betas=[hps.train.betas[0], hps.train.betas[1]],
                                 eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate,
                                 betas=[hps.train.betas[0], hps.train.betas[1]],
                                 eps=hps.train.eps)

    if use_ddp:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    # Load pretrained baseline if specified
    if hps.causal.weight_path:
        checkpoint = torch.load(hps.causal.weight_path, map_location=f'cuda:{rank}',
                                weights_only=False)
        model_to_load = net_g.module if use_ddp else net_g
        model_to_load.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f"Loaded pretrained weights from {hps.causal.weight_path}")

    # Try resume from checkpoint
    epoch_str = 1
    ckpt_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
    if ckpt_path:
        _, _, _, epoch_str = utils.load_checkpoint(ckpt_path, net_g, optim_g)
        ckpt_d = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        if ckpt_d:
            utils.load_checkpoint(ckpt_d, net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = torch.amp.GradScaler('cuda', enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d],
                           [scheduler_g, scheduler_d], scaler,
                           [train_loader, eval_loader, causal_loader],
                           logger, [writer, writer_eval], use_ddp)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, use_ddp=False):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader, causal_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    fp16_run = hps.train.fp16_run

    global global_step
    net_g.train()
    net_d.train()

    # Create causal data iterator
    causal_iter = iter(causal_loader)
    use_causal = (epoch >= hps.causal.start_epoch and hps.causal.reg_type != 'none')

    for batch_idx, items in enumerate(train_loader):
        # ==================== REAL DATA STREAM ====================
        if hps.model.use_spk:
            c, spec, y, spk, f0_src, f0_tgt = items
            g = spk.cuda(rank, non_blocking=True)
        else:
            c, spec, y, spk, f0_src, f0_tgt = items
            g = None

        spec = spec.cuda(rank, non_blocking=True)
        y = y.cuda(rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        f0_tgt = f0_tgt.cuda(rank, non_blocking=True)

        mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)

        with torch.amp.autocast('cuda', enabled=fp16_run):
            y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = \
                net_g(c, spec, g=g, f0=f0_tgt, mel=mel)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax)

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # Discriminator step
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with torch.amp.autocast('cuda', enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        # Generator step
        with torch.amp.autocast('cuda', enabled=fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with torch.amp.autocast('cuda', enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        # ==================== CAUSAL DATA STREAM ====================
        loss_causal = torch.tensor(0.0).cuda(rank)
        loss_contrastive = torch.tensor(0.0).cuda(rank)
        loss_ranking = torch.tensor(0.0).cuda(rank)

        if use_causal:
            try:
                causal_batch = next(causal_iter)
            except StopIteration:
                causal_iter = iter(causal_loader)
                causal_batch = next(causal_iter)

            causal_features, causal_effects, data_splits = causal_batch
            causal_features = causal_features.cuda(rank)
            causal_effects = [ce.cuda(rank) for ce in causal_effects]

            # Forward through content encoder + causal projector
            model_g = net_g.module if use_ddp else net_g
            with torch.amp.autocast('cuda', enabled=fp16_run):
                causal_embeds = model_g.get_causal_embeddings(causal_features)

            # Compute causal losses
            with torch.amp.autocast('cuda', enabled=False):
                if hps.causal.reg_type in ['contrastive', 'both']:
                    loss_contrastive = calc_contrastive_loss(
                        causal_embeds, causal_effects, data_splits,
                        contrastive_weight=hps.causal.lambda_contrastive,
                        tau=hps.causal.tau,
                        non_causal_thresh=hps.causal.non_causal_thresh,
                        causal_thresh=hps.causal.causal_thresh
                    )

                if hps.causal.reg_type in ['ranking', 'both']:
                    loss_ranking = calc_ranking_loss(
                        causal_embeds, causal_effects, data_splits,
                        ranking_weight=hps.causal.lambda_ranking,
                        margin=hps.causal.ranking_margin
                    )

                loss_causal = loss_contrastive + loss_ranking

            # Add causal loss to generator loss
            loss_gen_all = loss_gen_all + loss_causal

        # Backprop generator
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        # ==================== LOGGING ====================
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch, 100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "loss/g/mel": loss_mel,
                    "loss/g/kl": loss_kl,
                    "loss/g/fm": loss_fm,
                    "loss/causal/total": loss_causal,
                    "loss/causal/contrastive": loss_contrastive,
                    "loss/causal/ranking": loss_ranking,
                    "learning_rate": lr,
                    "grad_norm_g": grad_norm_g,
                }
                summarize(writer=writer, global_step=global_step, scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, use_ddp)
                utils.save_checkpoint(net_g, optim_g, lr, epoch,
                                      os.path.join(hps.model_dir, f"G_{global_step}.pth"))
                utils.save_checkpoint(net_d, optim_d, lr, epoch,
                                      os.path.join(hps.model_dir, f"D_{global_step}.pth"))

        global_step += 1

    if rank == 0:
        logger.info(f'====> Epoch: {epoch}')


def summarize(writer, global_step, scalars=None, images=None, audios=None, audio_sampling_rate=16000):
    """Write summaries to tensorboard."""
    if scalars:
        for k, v in scalars.items():
            writer.add_scalar(k, v, global_step)
    if images:
        for k, v in images.items():
            writer.add_image(k, v, global_step, dataformats='HWC')
    if audios:
        for k, v in audios.items():
            writer.add_audio(k, v, global_step, sample_rate=audio_sampling_rate)


def evaluate(hps, generator, eval_loader, writer_eval, use_ddp=False):
    """Standard VC evaluation (mel reconstruction quality)."""
    generator.eval()

    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, spec, y, spk, f0_src, f0_tgt = items
            g = spk[:1].cuda(0) if hps.model.use_spk else None
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            f0_tgt = f0_tgt[:1].cuda(0)
            break

        mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
        model_g = generator.module if use_ddp else generator
        y_hat = model_g.infer(c, g=g, f0=f0_tgt, mel=mel)

        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels,
            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            hps.data.mel_fmin, hps.data.mel_fmax)

    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
        "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
    }
    audio_dict = {"gen/audio": y_hat[0], "gt/audio": y[0]}
    summarize(writer=writer_eval, global_step=global_step,
              images=image_dict, audios=audio_dict,
              audio_sampling_rate=hps.data.sampling_rate)
    generator.train()


if __name__ == "__main__":
    main()
