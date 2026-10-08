from __future__ import annotations

import argparse
import logging
import math
import os
import pickle as pkl
import random
import re
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import metrics
from .config import (
    load_args_from_file,
    validate_train_args,
)
from .constants import (
    IGBP_CLASS_NAMES,
    LOSS_TYPES,
    NORMALIZATION_MODES,
    SPECIAL_VALUES,
    TRAIN_DATASET_TYPES,
    TrainArgDefaults,
)
from .datasets import DFC2020, SEN12MS, BaseDataset
from .logging_utils import log_args_config, setup_logging
from .losses import create_loss_function
from .losses_oneformer import OneFormerHungarianLoss
from .model_factory import create_lr_scheduler, create_model, create_optimizer, setup_seeds
from .profiling_utils import TrainingProfiler, create_profiler_from_args
from .result_saver import ConfigSaver, MetricsLogger
from .sampler import ResumableRandomSampler
from .tensorboard import TensorBoardLogger, create_tensorboard_logger
from .timing_utils import (
    Timer,
    get_aggregator,
    load_timing_config_from_args,
    reset_aggregator,
    set_timing_args,
)
from .training_utils import GradualUnfreezer, NoisyLabelScoreEMA, build_checkpoint_payload
from .utils import (
    display_input_batch,
    display_label_batch,
    get_dfc2020_subset,
    log_and_raise,
    make_training_result,
    make_training_state,
    make_validation_metrics,
    training_state_items,
    validate_string_value,
    validation_metrics_items,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        args: argparse.Namespace,
        metrics_logger: MetricsLogger | None = None,
        tb_logger: TensorBoardLogger | None = None,
    ) -> None:
        self.args = args
        self.metrics_logger = metrics_logger
        self.tb_logger = tb_logger
        self.enable_noisy_label_weighting: bool = bool(
            getattr(args, "enable_noisy_label_weighting", TrainArgDefaults["enable_noisy_label_weighting"])
        )
        self.noisy_label_score_ema = NoisyLabelScoreEMA()
        self.noisy_label_warmup_steps: int = int(getattr(args, "noisy_label_warmup_steps", TrainArgDefaults["noisy_label_warmup_steps"]))
        self.noisy_label_ema_alpha: float = float(getattr(args, "noisy_label_ema_alpha", TrainArgDefaults["noisy_label_ema_alpha"]))
        self.noisy_label_min_weight: float = float(getattr(args, "noisy_label_min_weight", TrainArgDefaults["noisy_label_min_weight"]))
        self.noisy_label_loss_compatible: bool = False
        self.loss_type: str = validate_string_value(
            getattr(args, "loss_type", TrainArgDefaults["loss_type"]),
            LOSS_TYPES,
            "loss type",
        )

        if self.enable_noisy_label_weighting:
            supported_losses = {"cross_entropy", "label_smoothing_cross_entropy"}
            self.noisy_label_loss_compatible = self.loss_type in supported_losses
            if not self.noisy_label_loss_compatible:
                logger.warning(
                    "Noisy-label weighting currently supports only cross_entropy and label_smoothing_cross_entropy. Disabling for loss_type=%s.",
                    self.loss_type,
                )
                self.enable_noisy_label_weighting = False
            else:
                logger.info(
                    "Noisy-label weighting enabled: warmup_steps=%d, ema_alpha=%.3f, min_weight=%.3f",
                    self.noisy_label_warmup_steps,
                    self.noisy_label_ema_alpha,
                    self.noisy_label_min_weight,
                )

    def _extract_batch_ids(self, batch: Mapping[str, Any], batch_size: int, global_step: int) -> list[str]:
        raw_ids = batch.get("id")
        if isinstance(raw_ids, (list, tuple)) and len(raw_ids) == batch_size:
            return [str(sample_id) for sample_id in raw_ids]
        return [f"step{global_step}_sample{i}" for i in range(batch_size)]

    def _extract_copernicus_metadata(self, batch: Mapping[str, Any], use_gpu: bool) -> dict[str, torch.Tensor] | None:
        metadata: dict[str, torch.Tensor] = {}
        for key in ("coord", "lon", "lat", "area", "time"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                metadata[key] = value.cuda() if use_gpu else value

        if "coord" not in metadata and "lon" in metadata and "lat" in metadata:
            lon = metadata["lon"].view(metadata["lon"].shape[0], -1)[:, 0]
            lat = metadata["lat"].view(metadata["lat"].shape[0], -1)[:, 0]
            metadata["coord"] = torch.stack([lon, lat], dim=-1)

        return metadata if metadata else None

    def _resolve_oneformer_class_names(self, n_classes: int) -> list[str]:
        class_names = getattr(self.args, "class_names", None)
        if isinstance(class_names, (list, tuple)) and class_names:
            return [str(name) for name in class_names]
        if n_classes == 10:
            return list(IGBP_CLASS_NAMES["ALL_CLASSES"])
        if n_classes == 9:
            return list(IGBP_CLASS_NAMES["NO_SAVANNA"])
        return [f"Class {i}" for i in range(n_classes)]

    @staticmethod
    def _get_oneformer_outputs(outputs: Any) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        class_logits = getattr(outputs, "class_queries_logits", None)
        if class_logits is None:
            class_logits = getattr(outputs, "transformer_decoder_class_predictions", None)
        mask_logits = getattr(outputs, "masks_queries_logits", None)
        if mask_logits is None:
            mask_logits = getattr(outputs, "transformer_decoder_mask_predictions", None)
        query_emb = getattr(outputs, "transformer_decoder_contrastive_queries", None)
        if query_emb is None:
            query_emb = getattr(outputs, "contrastive_queries", None)
        text_emb = getattr(outputs, "text_queries", None)
        if text_emb is None:
            text_emb = getattr(outputs, "text_embeddings", None)
        return class_logits, mask_logits, query_emb, text_emb

    @staticmethod
    def _compute_oneformer_query_targets(
        mask_logits: torch.Tensor,
        target: torch.Tensor,
        n_classes: int,
    ) -> torch.Tensor:
        bsz, num_queries, height, width = mask_logits.shape
        scores = mask_logits.new_full((bsz, num_queries, n_classes), float("-inf"))
        for cls_idx in range(n_classes):
            class_mask = target == cls_idx
            if not torch.any(class_mask):
                continue
            class_mask = class_mask.unsqueeze(1)
            denom = class_mask.sum(dim=(2, 3)).clamp_min(1.0)
            score = (mask_logits * class_mask).sum(dim=(2, 3)) / denom
            has_pixels = class_mask.sum(dim=(2, 3)) > 0
            scores[:, :, cls_idx] = torch.where(has_pixels, score, scores[:, :, cls_idx])

        valid = torch.isfinite(scores).any(dim=-1)
        targets = scores.argmax(dim=-1)
        targets = torch.where(valid, targets, torch.full_like(targets, -1))
        return targets

    @staticmethod
    def _compute_oneformer_contrastive_loss(
        query_emb: torch.Tensor | None,
        text_emb: torch.Tensor | None,
        query_targets: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor | None:
        if query_emb is None or text_emb is None:
            return None
        if temperature <= 0:
            return None

        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(0).expand(query_emb.shape[0], -1, -1)

        if text_emb.shape[0] != query_emb.shape[0]:
            return None

        q_norm = F.normalize(query_emb, dim=-1)
        t_norm = F.normalize(text_emb, dim=-1)
        logits = torch.einsum("bqd,btd->bqt", q_norm, t_norm) / temperature
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), query_targets.reshape(-1), ignore_index=-1)
        return loss

    def _update_noisy_label_ema(
        self,
        sample_ids: list[str],
        sample_losses: torch.Tensor,
    ) -> torch.Tensor:
        sample_losses_cpu = sample_losses.detach().cpu().tolist()
        ema_scores: list[float] = []
        for sample_id, sample_loss in zip(sample_ids, sample_losses_cpu, strict=False):
            old_value = self.noisy_label_score_ema.get(sample_id, sample_loss)
            new_value = (self.noisy_label_ema_alpha * old_value) + ((1.0 - self.noisy_label_ema_alpha) * sample_loss)
            self.noisy_label_score_ema.set(sample_id, new_value)
            ema_scores.append(new_value)

        return torch.tensor(ema_scores, dtype=sample_losses.dtype, device=sample_losses.device)

    def _compute_noisy_label_weights(self, ema_scores: torch.Tensor, global_step: int) -> torch.Tensor:
        if global_step < self.noisy_label_warmup_steps:
            return torch.ones_like(ema_scores)

        score_min = ema_scores.min()
        score_max = ema_scores.max()
        score_range = (score_max - score_min).abs().item()
        if score_range < 1e-12:
            return torch.ones_like(ema_scores)

        normalized_scores = (ema_scores - score_min) / (score_max - score_min + 1e-12)
        weights = 1.0 - normalized_scores * (1.0 - self.noisy_label_min_weight)
        result: torch.Tensor = weights.clamp(min=self.noisy_label_min_weight, max=1.0)
        return result

    def _compute_weighted_noisy_label_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_ids: list[str],
        global_step: int,
    ) -> tuple[torch.Tensor, float, float]:
        label_smoothing = (
            float(getattr(self.args, "label_smoothing", TrainArgDefaults["label_smoothing"]))
            if self.loss_type == "label_smoothing_cross_entropy"
            else 0.0
        )
        pixel_loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=SPECIAL_VALUES["IGNORE_LABEL"],
            reduction="none",
            label_smoothing=label_smoothing,
        )
        valid_mask = target != SPECIAL_VALUES["IGNORE_LABEL"]
        valid_mask_float = valid_mask.to(pixel_loss.dtype)

        sample_valid_pixels = valid_mask_float.view(valid_mask.shape[0], -1).sum(dim=1)
        sample_loss = (pixel_loss * valid_mask_float).view(valid_mask.shape[0], -1).sum(dim=1)
        sample_loss = sample_loss / sample_valid_pixels.clamp_min(1.0)
        sample_loss = torch.where(
            sample_valid_pixels > 0,
            sample_loss,
            torch.zeros_like(sample_loss),
        )

        ema_scores = self._update_noisy_label_ema(sample_ids, sample_loss)
        sample_weights = self._compute_noisy_label_weights(ema_scores, global_step)
        sample_weights = torch.where(
            sample_valid_pixels > 0,
            sample_weights,
            torch.zeros_like(sample_weights),
        )

        weight_sum = sample_weights.sum().clamp_min(1.0)
        weighted_loss = (sample_loss * sample_weights).sum() / weight_sum

        valid_samples = sample_valid_pixels > 0
        mean_weight = float(sample_weights[valid_samples].mean().item()) if valid_samples.any() else 0.0
        mean_score = float(ema_scores.mean().item()) if ema_scores.numel() > 0 else 0.0
        return weighted_loss, mean_score, mean_weight

    # main training function (trains for one epoch)
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        aux_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
        optimizer: torch.optim.Optimizer,
        writer: SummaryWriter,
        step: int = 0,
        num_steps_limit: int | None = None,
        accumulation_steps: int | None = None,
        dataloader_generator: torch.Generator | None = None,
        train_sampler: ResumableRandomSampler | None = None,
        enable_timing: bool = False,
        current_epoch: int | None = None,
        profiler: TrainingProfiler | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        gradual_unfreezer: GradualUnfreezer | None = None,
        oneformer_hungarian_loss: OneFormerHungarianLoss | None = None,
    ) -> tuple[nn.Module, int]:
        model.train()
        global_step = step

        oneformer_contrastive_weight = float(getattr(self.args, "oneformer_contrastive_weight", 0.0))
        oneformer_contrastive_temperature = float(getattr(self.args, "oneformer_contrastive_temperature", 0.07))
        oneformer_contrastive_use_gt = bool(getattr(self.args, "oneformer_contrastive_use_gt", True))

        ds = cast(BaseDataset, train_loader.dataset)
        no_savanna: bool = ds.no_savanna
        display_channels: tuple[int, ...] | int = ds.display_channels
        brightness_factor: int = ds.brightness_factor

        tqdm_duration = len(train_loader) if num_steps_limit is None else min(len(train_loader), num_steps_limit - step)
        pbar = tqdm(total=tqdm_duration, desc="[Train]")

        # Wrap entire training epoch in timer
        with Timer(
            "training/epoch",
            logger,
            writer,
            global_step,
            component="training",
            enable_timing=enable_timing,
        ):
            for i, batch in enumerate(train_loader):
                abs_batch_idx = train_sampler._current_idx if train_sampler is not None else i
                if gradual_unfreezer is not None and current_epoch is not None:
                    gradual_unfreezer.maybe_apply(
                        model=model,
                        current_epoch=current_epoch,
                        current_step=global_step,
                        writer=writer,
                    )

                # Profile the core training step (forward, backward, optimizer)
                with profiler.profile_step(global_step) if profiler else nullcontext():
                    image, target = batch["image"], batch["label"]
                    sample_ids = self._extract_batch_ids(
                        batch=batch,
                        batch_size=target.shape[0],
                        global_step=global_step,
                    )
                    optimizer.zero_grad()

                    if self.args.use_gpu:
                        image, target = image.cuda(), target.cuda()

                    oneformer_contrastive_loss: torch.Tensor | None = None
                    prediction: torch.Tensor | None = None
                    aux_prediction: torch.Tensor | None = None
                    noisy_mean_score: float | None = None
                    noisy_mean_weight: float | None = None

                    model_ref = cast(nn.Module, model.module) if isinstance(model, nn.DataParallel) else model
                    use_oneformer_contrastive = (
                        getattr(self.args, "model", "") == "oneformer"
                        and getattr(model_ref, "full_oneformer", False)
                        and oneformer_contrastive_weight > 0.0
                    )

                    if use_oneformer_contrastive:
                        from .models.oneformer import OneFormer

                        oneformer_ref = cast(OneFormer, model_ref)
                        class_names = self._resolve_oneformer_class_names(getattr(model_ref, "n_classes", 0))
                        text_inputs = oneformer_ref.build_text_inputs(class_names, device=image.device)
                        if text_inputs.dim() == 2:
                            text_inputs = text_inputs.unsqueeze(0).expand(image.shape[0], -1, -1)
                        model_output, oneformer_outputs, oneformer_mask_logits = cast(
                            tuple[torch.Tensor, Any, Any],
                            oneformer_ref.forward_full(image, text_inputs=text_inputs),  # type: ignore[operator]
                        )
                        class_logits, _mask_logits, query_emb, text_emb = self._get_oneformer_outputs(oneformer_outputs)

                        query_targets: torch.Tensor | None
                        if oneformer_contrastive_use_gt:
                            query_targets = self._compute_oneformer_query_targets(
                                oneformer_mask_logits,
                                target,
                                len(class_names),
                            )
                        else:
                            query_targets = class_logits.argmax(dim=-1) if class_logits is not None else None

                        if query_targets is not None:
                            oneformer_contrastive_loss = self._compute_oneformer_contrastive_loss(
                                query_emb,
                                text_emb,
                                query_targets,
                                oneformer_contrastive_temperature,
                            )
                    elif getattr(self.args, "model", "") == "copernicus_fm":
                        metadata_batch = self._extract_copernicus_metadata(batch, use_gpu=bool(self.args.use_gpu))
                        model_output = model(image, metadata=metadata_batch)
                    else:
                        model_output = model(image)

                    # Handle auxiliary head output (dict or tuple)
                    if isinstance(model_output, dict):
                        prediction = model_output["main"]
                        aux_prediction = model_output.get("auxiliary")
                    elif isinstance(model_output, tuple):
                        prediction = model_output[0]
                        aux_prediction = model_output[1]
                    else:
                        prediction = model_output

                    # OneFormer: use Hungarian matching loss when training dict contains query outputs
                    if (
                        oneformer_hungarian_loss is not None
                        and isinstance(model_output, dict)
                        and "class_logits" in model_output
                        and "mask_logits" in model_output
                    ):
                        loss = oneformer_hungarian_loss(
                            model_output["class_logits"],
                            model_output["mask_logits"],
                            target,
                            auxiliary_predictions=model_output.get("auxiliary_predictions"),
                        )
                    elif self.enable_noisy_label_weighting and self.noisy_label_loss_compatible:
                        loss, noisy_mean_score, noisy_mean_weight = self._compute_weighted_noisy_label_loss(
                            prediction=prediction,
                            target=target,
                            sample_ids=sample_ids,
                            global_step=global_step,
                        )
                    else:
                        loss = loss_fn(prediction, target)

                    # Add auxiliary loss if present
                    if aux_prediction is not None:
                        aux_loss_impl = aux_loss_fn if aux_loss_fn is not None else loss_fn
                        aux_loss = aux_loss_impl(aux_prediction, target)
                        aux_weight = getattr(model.module if isinstance(model, nn.DataParallel) else model, "auxiliary_loss_weight", 0.4)
                        loss = loss + aux_weight * aux_loss

                    if oneformer_contrastive_loss is not None:
                        loss = loss + (oneformer_contrastive_weight * oneformer_contrastive_loss)
                    loss.backward()

                    if accumulation_steps is None or (abs_batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()

                        # Step LR scheduler
                        if scheduler is not None:
                            scheduler.step()

                        optimizer.zero_grad(set_to_none=True)

                if num_steps_limit is not None and global_step >= num_steps_limit:
                    pbar.update()
                    break

                writer.add_scalar("train/loss", loss, global_step=global_step)

                if oneformer_contrastive_loss is not None:
                    writer.add_scalar("train/oneformer_contrastive_loss", oneformer_contrastive_loss, global_step=global_step)
                if oneformer_hungarian_loss is not None and isinstance(model_output, dict) and "class_logits" in model_output:
                    writer.add_scalar("train/oneformer_hungarian_loss", loss, global_step=global_step)
                if aux_prediction is not None:
                    writer.add_scalar("train/aux_loss", aux_loss, global_step=global_step)
                if noisy_mean_score is not None and noisy_mean_weight is not None:
                    writer.add_scalar(
                        "train/noisy_label_mean_score",
                        noisy_mean_score,
                        global_step=global_step,
                    )
                    writer.add_scalar(
                        "train/noisy_label_mean_weight",
                        noisy_mean_weight,
                        global_step=global_step,
                    )

                # Log to metrics logger for CSV/JSON export
                if self.metrics_logger is not None:
                    self.metrics_logger.log_train_step(step=global_step, loss=loss.item(), epoch=current_epoch)

                # Extended TensorBoard logging (learning rate, gradient norm)
                if self.tb_logger is not None:
                    # Log learning rate
                    if getattr(self.args, "log_learning_rate", TrainArgDefaults["log_learning_rate"]):
                        self.tb_logger.log_learning_rate(optimizer, global_step)

                    # Log gradient norm
                    if getattr(self.args, "log_gradient_norm", TrainArgDefaults["log_gradient_norm"]):
                        self.tb_logger.log_gradient_norm(model, global_step)

                    # Log weight histograms (less frequently due to overhead)
                    if (
                        getattr(
                            self.args,
                            "log_weight_histograms",
                            False,
                        )
                        and global_step > 0
                        and global_step % (self.args.log_freq * 5) == 0
                    ):
                        self.tb_logger.log_weight_histograms(model, global_step)

                    # Log gradient histograms (less frequently due to overhead)
                    if (
                        getattr(
                            self.args,
                            "log_gradient_histograms",
                            False,
                        )
                        and global_step > 0
                        and global_step % (self.args.log_freq * 5) == 0
                    ):
                        self.tb_logger.log_gradient_histograms(model, global_step)

                # write some example images to tensorboard every n steps
                if global_step > 0 and global_step % self.args.log_freq == 0 and prediction is not None:
                    with Timer(
                        "training/tb_images",
                        logger,
                        writer,
                        global_step,
                        component="training",
                        aggregate=True,
                    ):
                        writer.add_images("train/input", image[:, 0:3, :, :], global_step=global_step)
                        writer.flush()

                        imgs = display_input_batch(image, display_channels, brightness_factor=brightness_factor)
                        writer.add_images("train/input", imgs, global_step=global_step)

                        # show predictions
                        imgs_np = display_label_batch(prediction, no_savanna=no_savanna)
                        imgs = torch.from_numpy(imgs_np)
                        writer.add_images("train/prediction", imgs, global_step=global_step)

                        # show ground-truth labels
                        imgs_np = display_label_batch(target, no_savanna=no_savanna)
                        imgs = torch.from_numpy(imgs_np)
                        writer.add_images("train/ground_truth", imgs, global_step=global_step)

                # run validation
                if global_step > 0 and global_step % self.args.val_freq == 0:
                    self.val(
                        model,
                        val_loader,
                        global_step,
                        loss_fn,
                        writer,
                        enable_timing=enable_timing,
                    )

                # save checkpoint (save_freq=0 disables intermediate checkpoints)
                if self.args.save_freq > 0 and global_step > 0 and global_step % self.args.save_freq == 0:
                    with Timer(
                        "training/checkpoint_save",
                        logger,
                        writer,
                        global_step,
                        component="training",
                    ):
                        self.export_model(
                            model,
                            optimizer=optimizer,
                            step=global_step,
                            dataloader_generator=dataloader_generator,
                            train_sampler=train_sampler,
                        )

                global_step += 1

                if global_step % self.args.log_batch_freq == 0:
                    logger.info(f"Step {global_step}, Loss: {loss:.4f}")
                pbar.set_description(f"[Train] Loss: {round(loss.item(), 4):.4f}")
                pbar.update()

        pbar.close()
        writer.flush()
        return (model, global_step)

    # main validation function (validates current model)
    def val(
        self,
        model: nn.Module,
        dataloader: DataLoader[Any],
        step: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        writer: SummaryWriter,
        enable_timing: bool = False,
    ) -> dict[str, float]:
        # set model to evaluation mode
        model.eval()

        # main validation loop
        pbar = tqdm(total=len(dataloader), desc="[Val]")
        loss = 0.0
        conf_mat = metrics.ConfMatrix(cast(BaseDataset, dataloader.dataset).n_classes)

        # Wrap entire validation epoch in timer
        with Timer(
            "training/val_epoch",
            logger,
            writer,
            step,
            component="training",
            enable_timing=enable_timing,
        ):
            for i, batch in enumerate(dataloader):
                # unpack sample
                image, target = batch["image"], batch["label"]

                # move data to gpu if model is on gpu
                if self.args.use_gpu:
                    image, target = image.cuda(), target.cuda()

                model_output = model(image)
                # Handle dict/tuple output (defensive; model should be in eval mode)
                if isinstance(model_output, dict):
                    prediction = model_output["main"]
                elif isinstance(model_output, tuple):
                    prediction = model_output[0]
                else:
                    prediction = model_output

                batch_loss = loss_fn(prediction, target).cpu().item()

                # Check for NaN/Inf in validation loss - indicates model weights are corrupted
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.warning(
                        f"NaN/Inf detected in validation loss at val_step {i} (training step {step}). "
                        "Model weights may be corrupted. Returning NaN metrics for early termination."
                    )
                    pbar.close()
                    # Return NaN metrics to signal early termination
                    return make_validation_metrics(
                        loss=float("nan"),
                        AA=float("nan"),
                        mIoU=float("nan"),
                        accuracy=float("nan"),
                        mean_f1=float("nan"),
                    )

                loss += batch_loss

                # calculate error metrics
                conf_mat.add_batch(target, prediction.max(1)[1])

                # update progressbar
                pbar.update()

                if i % self.args.log_batch_freq == 0:
                    logger.info(f"Val Step {i}, Loss: {loss / (i + 1):.4f}")

        # Calculate all metrics
        epoch_loss = loss / len(dataloader)
        aa = conf_mat.get_aa()
        miou = conf_mat.get_mIoU()
        accuracy = conf_mat.get_accuracy()
        mean_f1 = conf_mat.get_mean_f1()

        # write validation metrics to tensorboard
        writer.add_scalar(self.args.dataset_val + "/epoch_loss", epoch_loss, global_step=step)
        writer.add_scalar(self.args.dataset_val + "/AA", aa, global_step=step)
        writer.add_scalar(self.args.dataset_val + "/mIoU", miou, global_step=step)
        writer.add_scalar(self.args.dataset_val + "/accuracy", accuracy, global_step=step)
        writer.add_scalar(self.args.dataset_val + "/mean_f1", mean_f1, global_step=step)

        # Log to metrics logger for CSV/JSON export
        val_result = make_validation_metrics(
            loss=epoch_loss,
            AA=aa,
            mIoU=miou,
            accuracy=accuracy,
            mean_f1=mean_f1,
        )
        if self.metrics_logger is not None:
            self.metrics_logger.log_val_step(step, validation_metrics_items(val_result))

        # Extended TensorBoard logging (confusion matrix, per-class metrics)
        if self.tb_logger is not None:
            # Log confusion matrix as image
            cm_array = conf_mat.overall_conf_mat
            # Use proper IGBP class names for visualizations
            if conf_mat.n_classes == 10:
                class_names = list(IGBP_CLASS_NAMES["ALL_CLASSES"])
            elif conf_mat.n_classes == 9:
                class_names = list(IGBP_CLASS_NAMES["NO_SAVANNA"])
            else:
                class_names = [f"Class {i}" for i in range(conf_mat.n_classes)]
            self.tb_logger.log_confusion_matrix(cm_array, class_names, step, tag=f"{self.args.dataset_val}/confusion_matrix")

            # Log per-class IoU and F1 scores as bar charts
            per_class_iou = conf_mat.get_per_class_iou()
            per_class_f1 = conf_mat.get_per_class_f1()
            if per_class_iou is not None and per_class_f1 is not None:
                per_class_metrics = {
                    "IoU": per_class_iou.tolist(),
                    "F1": per_class_f1.tolist(),
                }
                self.tb_logger.log_per_class_metrics(
                    per_class_metrics,
                    class_names,
                    step,
                    tag_prefix=f"{self.args.dataset_val}/per_class",
                )

            # Log activation distributions during validation
            if getattr(self.args, "log_distributions", True) and step > 0:
                # Use first batch for activation logging
                first_batch = next(iter(dataloader))
                sample_input = first_batch["image"]
                if self.args.use_gpu:
                    sample_input = sample_input.cuda()
                self.tb_logger.log_activation_distributions(model, sample_input, step, tag=f"{self.args.dataset_val}/activations")

        # close progressbar
        logger.info(f"Validation Loss: {epoch_loss:.4f}")
        logger.info(f"Validation AA: {aa * 100:.2f}%")
        logger.info(f"Validation mIoU: {miou * 100:.2f}%")
        logger.info(f"Validation Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"Validation Mean F1: {mean_f1 * 100:.2f}%")
        pbar.set_description(f"[Val] AA: {conf_mat.get_aa() * 100:.2f}%")
        pbar.close()

        writer.flush()
        model.train()

        return val_result

    def export_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        name: str | None = None,
        step: int | None = None,
        dataloader_generator: torch.Generator | None = None,
        train_sampler: ResumableRandomSampler | None = None,
    ) -> None:
        # set output filename
        if name is not None:
            out_file = name
        else:
            out_file = "checkpoint"
            if step is not None:
                out_file += "_step_" + str(step)
        out_file = os.path.join(self.args.checkpoint_dir, out_file + ".pth")

        model_to_save = model.module if isinstance(model, nn.DataParallel) else model

        state = make_training_state(
            model_state=model_to_save.state_dict(),
            step=step,
            optimizer_state=optimizer.state_dict() if optimizer is not None else None,
            rng_state_torch=torch.get_rng_state(),
            rng_state_numpy=cast(tuple[Any, ...], np.random.get_state()),
            rng_state_python=random.getstate(),
            rng_state_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            rng_state_generator=dataloader_generator.get_state() if dataloader_generator is not None else None,
            sampler_state=train_sampler.get_state() if train_sampler is not None else None,
        )
        payload = build_checkpoint_payload(training_state_items(state))
        torch.save(payload, out_file)


def run_training(
    args_file_path: str,
    args_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run training with programmatic arguments.

    Args:
        args_file_path: Path to combined arguments file (common + model-specific)
        args_override: Optional mapping to override specific arguments

    Returns:
        Training result summary
    """
    # Load all arguments (common + model-specific) from a single file
    args = load_args_from_file(args_file_path, TrainArgDefaults.copy(), use_model_defaults=True)

    # Apply overrides if provided
    if args_override:
        for key, value in args_override.items():
            setattr(args, key, value)

    # Validate all arguments early with clear error messages
    validate_train_args(args)

    # Make args.enable_timing available to timing decorators
    set_timing_args(args)
    load_timing_config_from_args(args)
    reset_aggregator()  # Reset aggregator for fresh run

    return _run_training_with_args(args)


def _run_training_with_args(args: argparse.Namespace) -> dict[str, Any]:
    """Internal function to run training with loaded arguments."""
    # set up logging using config file
    log_dir_path = os.path.join(args.log_dir, args.experiment_name)
    if os.path.exists(log_dir_path):
        if not args.continue_learning_from_previous:
            log_and_raise(ValueError, f"log_dir {log_dir_path} already exists!", logger_instance=logger)
        else:
            logger.info(f"Continuing training using existing log_dir {log_dir_path}")
    else:
        os.makedirs(log_dir_path, exist_ok=True)

    log_file_path = os.path.join(log_dir_path, "logs.log")

    setup_logging(str(args.log_level), log_file_path)

    log_args_config(args, logger=logger, header="Config", blank_line=True)

    # fix seeds and set pytorch to deterministic mode
    if args.seed is not None:
        args.seed = int(args.seed)
    seed_worker, g = setup_seeds(args.seed)

    # set flags for GPU processing if available
    if torch.cuda.is_available():
        args.use_gpu = True
        args.num_gpus = torch.cuda.device_count()
        if args.num_gpus > 1:
            logger.info(f"Using {args.num_gpus} GPUs for training")
    else:
        args.use_gpu = False
        args.num_gpus = 0

    # create checkpoint dir early (needed for checkpoint loading)
    args.checkpoint_dir = os.path.join(args.log_dir, args.experiment_name, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize variables for checkpoint restoration
    start_epoch = 0
    step = 0
    ckpt = None
    sampler_state = None

    # Checkpoint load must precede dataset/dataloader creation so they see the restored RNG state
    if args.continue_learning_from_previous:
        # remove any final checkpoint files
        try:
            for fn in os.listdir(args.checkpoint_dir):
                if fn.startswith("final") and fn.endswith(".pth"):
                    fp = os.path.join(args.checkpoint_dir, fn)
                    try:
                        os.remove(fp)
                        logger.info(f"Removed final checkpoint {fp}")
                    except Exception:
                        logger.warning(f"Could not remove final checkpoint {fp}")
        except Exception:
            pass

        # find latest checkpoint
        pattern = re.compile(r"checkpoint_step_(\d+)\.pth$")
        candidates = []
        try:
            for fn in os.listdir(args.checkpoint_dir):
                m = pattern.match(fn)
                if m:
                    try:
                        num = int(m.group(1))
                        candidates.append((num, os.path.join(args.checkpoint_dir, fn)))
                    except Exception:
                        continue
        except Exception:
            pass

        if len(candidates) > 0:
            candidates.sort()
            last_step, last_path = candidates[-1]
            logger.info(f"Found checkpoint to continue from: {last_path} (step {last_step})")

            try:
                ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning("Failed to load checkpoint %s: %s", last_path, e)
                ckpt = None

            if ckpt is not None:
                step = int(ckpt.get("step", last_step)) + 1

                if "rng_state_torch" in ckpt:
                    torch.set_rng_state(ckpt["rng_state_torch"].cpu().byte())
                if "rng_state_numpy" in ckpt:
                    np.random.set_state(ckpt["rng_state_numpy"])
                if "rng_state_python" in ckpt:
                    random.setstate(ckpt["rng_state_python"])
                if args.use_gpu and "rng_state_cuda" in ckpt:
                    cuda_states = [s.cpu().byte() for s in ckpt["rng_state_cuda"]]
                    torch.cuda.set_rng_state_all(cuda_states)
                if "rng_state_generator" in ckpt:
                    g.set_state(ckpt["rng_state_generator"].cpu().byte())

                sampler_state = ckpt.get("sampler_state")

    # Get normalization mode
    normalization_mode = validate_string_value(
        getattr(args, "normalization_mode", TrainArgDefaults["normalization_mode"]),
        NORMALIZATION_MODES,
        "normalization mode",
    )

    # Create base training dataset
    train_set_base = SEN12MS(
        args.data_dir_train,
        subset="train",
        no_savanna=args.no_savanna,
        use_s2hr=args.use_s2hr,
        use_s2mr=args.use_s2mr,
        use_s2lr=args.use_s2lr,
        use_s1=args.use_s1,
        normalization_mode=normalization_mode,
        border_exclusion=args.border_exclusion,
        border_exclusion_size=args.border_exclusion_size,
    )
    n_classes: int = train_set_base.n_classes
    n_inputs: int = train_set_base.n_inputs

    train_set: SEN12MS | Any = train_set_base

    dataset_type = validate_string_value(args.dataset_val, TRAIN_DATASET_TYPES, "validation dataset type")
    val_set: SEN12MS | DFC2020
    if dataset_type == "sen12ms_holdout":
        val_set = SEN12MS(
            args.data_dir_train,
            subset="holdout",
            no_savanna=args.no_savanna,
            use_s2hr=args.use_s2hr,
            use_s2mr=args.use_s2mr,
            use_s2lr=args.use_s2lr,
            use_s1=args.use_s1,
            normalization_mode=normalization_mode,
        )
    else:
        # DFC2020 validation or test set
        dfc2020_subset = get_dfc2020_subset(dataset_type)
        val_set = DFC2020(
            args.data_dir_val,
            subset=dfc2020_subset,
            no_savanna=args.no_savanna,
            use_s2hr=args.use_s2hr,
            use_s2mr=args.use_s2mr,
            use_s2lr=args.use_s2lr,
            use_s1=args.use_s1,
            normalization_mode=normalization_mode,
        )

    train_sampler = ResumableRandomSampler(train_set, generator=g)

    # Restore sampler state if resuming
    if sampler_state is not None:
        train_sampler.set_state(sampler_state)
        logger.info("Restored sampler state, starting from index %d", sampler_state.get("start_idx", 0))

    # num_workers=0 gives full determinism; otherwise keep workers but disable persistent_workers
    train_loader: DataLoader[Mapping[str, Any]] = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,  # Use custom sampler instead of shuffle=True
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,  # CHANGED: disable for determinism
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,  # CHANGED
        worker_init_fn=seed_worker,
    )

    if ckpt is not None:
        batches_per_epoch = len(train_set) // args.batch_size
        start_epoch = step // batches_per_epoch
        logger.info(f"Resuming from epoch {start_epoch}, step {step}")

    logger.info(f"{'num_channels':<20}  {n_inputs}")
    logger.info(f"{'num_classes':<20}  {n_classes}")

    # Create model
    model = create_model(args, n_classes, n_inputs)

    if args.use_gpu:
        if args.num_gpus > 1:
            model = nn.DataParallel(model)
        model = model.cuda()

    loss_fn = create_loss_function(args, ignore_index=SPECIAL_VALUES["IGNORE_LABEL"])
    loss_type = getattr(args, "loss_type", TrainArgDefaults["loss_type"])
    aux_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    if loss_type == "cross_entropy_dice":
        aux_loss_fn = nn.CrossEntropyLoss(ignore_index=SPECIAL_VALUES["IGNORE_LABEL"], reduction="mean")
    logger.info(f"Using loss function: {loss_type}")

    oneformer_hungarian_loss: OneFormerHungarianLoss | None = None
    if getattr(args, "model", "") == "oneformer":
        from .constants import OneFormerArgDefaults as _ofd

        oneformer_hungarian_loss = OneFormerHungarianLoss(
            num_classes=n_classes,
            num_queries=getattr(args, "oneformer_num_queries", _ofd["oneformer_num_queries"]),
            no_object_weight=getattr(args, "oneformer_no_object_weight", _ofd["oneformer_no_object_weight"]),
            class_weight=getattr(args, "oneformer_class_weight", _ofd["oneformer_class_weight"]),
            mask_weight=getattr(args, "oneformer_mask_weight", _ofd["oneformer_mask_weight"]),
            dice_weight=getattr(args, "oneformer_dice_weight", _ofd["oneformer_dice_weight"]),
            aux_loss_weight=getattr(args, "oneformer_aux_loss_weight", _ofd["oneformer_aux_loss_weight"]),
            ignore_index=SPECIAL_VALUES["IGNORE_LABEL"],
        )
        if bool(getattr(args, "use_gpu", False)):
            oneformer_hungarian_loss = oneformer_hungarian_loss.cuda()
        logger.info("OneFormer Hungarian matching loss created.")

    optimizer = create_optimizer(model, args, args)

    # Create LR scheduler
    scheduler = create_lr_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    if scheduler is not None:
        logger.info(f"Using LR scheduler: {args.lr_scheduler}")

    # Load model and optimizer state from checkpoint
    if ckpt is not None:
        with Timer("training/state_restore", logger, component="training"):
            try:
                model_to_load = model.module if isinstance(model, nn.DataParallel) else model
                model_to_load.load_state_dict(ckpt["model_state_dict"])
                logger.info("Restored model state")
            except Exception as e:
                logger.warning(f"Failed to load model state: {str(e)}")

            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if args.use_gpu:
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.cuda()
                    logger.info("Restored optimizer state")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {str(e)}")

        # Check if training is complete
        max_possible_steps = int(args.num_steps) if args.num_steps is not None else (args.max_epochs * len(train_loader))
        if step >= max_possible_steps:
            logger.info(f"Training already complete (step {step} >= {max_possible_steps})")
            ModelTrainer(args).export_model(model, optimizer, name="final", step=step)
            return make_training_result(
                log_dir=args.log_dir,
                experiment_name=args.experiment_name,
                checkpoint_dir=args.checkpoint_dir,
                completed=True,
                step=step,
            )

    # Set up tensorboard and save config
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.experiment_name))
    with open(os.path.join(args.checkpoint_dir, "args.pkl"), "wb") as f:
        pkl.dump(args, f)

    # Initialize metrics logger and save config
    metrics_logger = MetricsLogger(args.log_dir, args.experiment_name)
    ConfigSaver.save_args_summary(args, args.log_dir, args.experiment_name)

    # Initialize profiler if enabled
    profiler = create_profiler_from_args(args)

    # Initialize TensorBoard logger for extended logging
    tb_logger = create_tensorboard_logger(writer, args)

    # Log model graph to TensorBoard (once at start)
    if getattr(args, "enable_tensorboard_logging", TrainArgDefaults["enable_tensorboard_logging"]):
        with Timer("training/tb_init", logger, component="training"):
            sample_batch = next(iter(train_loader))
            sample_input = sample_batch["image"]
            if args.use_gpu:
                sample_input = sample_input.cuda()

            # Log model graph
            if getattr(args, "log_model_graph", TrainArgDefaults["log_model_graph"]):
                tb_logger.log_model_graph(model, sample_input)

            # Log sample input images
            if getattr(args, "log_batch_images", TrainArgDefaults["log_batch_images"]):
                tb_logger.log_batch_images(sample_input, step=0, tag="input_samples/initialization")

            # Log activation distributions
            if getattr(args, "log_distributions", TrainArgDefaults["log_distributions"]):
                tb_logger.log_activation_distributions(model, sample_input, step=0)
                tb_logger.log_layer_outputs(model, sample_input, step=0)

            # Log hyperparameters
            hparams_items = [(key, value) for key, value in vars(args).items() if not key.startswith("_")]
            tb_logger.log_hyperparameters(hparams_items)

            tb_logger.log_config_summary(hparams_items, step=0)

            # Log frozen layer information if applicable
            base_model: Any = cast(nn.Module, model.module) if isinstance(model, nn.DataParallel) else model
            if hasattr(base_model, "get_frozen_layer_info"):
                frozen_info = base_model.get_frozen_layer_info()
                if frozen_info["frozen_layers"] > 0:
                    # Log frozen layer count
                    writer.add_scalar(
                        "model/frozen_layers_count",
                        frozen_info["frozen_layers"],
                        global_step=0,
                    )
                    frozen_names_str = ", ".join(frozen_info["frozen_layer_names"])
                    writer.add_text(
                        "model/frozen_layer_names",
                        f"Frozen {frozen_info['frozen_layers']}/{frozen_info['total_layers']} layers: {frozen_names_str}",
                        global_step=0,
                    )
                    logger.info(f"TensorBoard: Logged frozen layer info - {frozen_info['frozen_layers']}/{frozen_info['total_layers']} layers frozen")

    # Training loop
    num_steps = int(args.num_steps) if args.num_steps is not None else None
    accumulation_steps = int(args.accumulation_steps) if args.accumulation_steps is not None else None
    trainer = ModelTrainer(args, metrics_logger=metrics_logger, tb_logger=tb_logger)
    gradual_unfreezer = GradualUnfreezer(args)

    for epoch in range(start_epoch, args.max_epochs):
        logger.info(f"====== Epoch {epoch + 1}/{args.max_epochs} ======")

        model, step = trainer.train(
            model,
            train_loader,
            val_loader,
            loss_fn,
            aux_loss_fn,
            optimizer,
            writer,
            step=step,
            num_steps_limit=num_steps,
            accumulation_steps=accumulation_steps,
            dataloader_generator=g,
            train_sampler=train_sampler,
            enable_timing=args.enable_timing,
            current_epoch=epoch,
            profiler=profiler,
            scheduler=scheduler,
            gradual_unfreezer=gradual_unfreezer,
            oneformer_hungarian_loss=oneformer_hungarian_loss,
        )

        if train_sampler is not None:
            train_sampler.reset()

        if num_steps is not None and step >= num_steps:
            logger.info(f"Reached num_steps={num_steps}, stopping training.")
            break

    trainer.export_model(model, optimizer, name="final", step=-1, dataloader_generator=g)

    # Finish profiling if enabled
    profiler.finish()

    # Save all metrics to CSV and JSON
    metrics_logger.save_all()
    logger.info("Training metrics saved to CSV and JSON")

    # Log timing summary if timing was enabled
    if args.enable_timing:
        aggregator = get_aggregator()
        aggregator.log_summary(logger, prefix="Training")
        aggregator.to_tensorboard(writer, step, prefix="timing_summary")

    writer.close()

    # Return experiment info for continuous run
    return make_training_result(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        completed=True,
        final_step=step,
    )


def main() -> dict[str, Any]:
    """Main entry point for command-line usage."""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--args_file_path",
        type=str,
        required=True,
        help="path to a combined args file (key=value per line) containing both common and model-specific parameters",
    )

    # parse only the path, then load rest from file
    parsed = parser.parse_args()

    return run_training(parsed.args_file_path)


if __name__ == "__main__":
    main()
