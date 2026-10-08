import logging
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ..constants import VISUALIZATION_DEFAULTS, TrainArgDefaults
from ..messages import TENSORBOARD_LOGGER_WARNINGS

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """Enhanced TensorBoard logger with support for various visualization types.

    Provides methods for logging:
    - Model architecture graphs
    - Weight and gradient histograms
    - Learning rate tracking
    - Confusion matrices
    - PR curves and ROC curves
    - Custom images and figures
    - Text summaries
    - Embeddings

    Attributes:
        writer: TensorBoard SummaryWriter instance
        enable_logging: Whether logging is enabled
        log_freq: Frequency of logging (in steps)
    """

    def __init__(
        self,
        writer: SummaryWriter,
        enable_logging: bool = True,
        log_freq: int = 100,
    ) -> None:
        """Initialize the TensorBoard logger.

        Args:
            writer: TensorBoard SummaryWriter instance
            enable_logging: Whether to enable TensorBoard logging
            log_freq: How often to log (in steps)
        """
        self.writer = writer
        self.enable_logging = enable_logging
        self.log_freq = log_freq
        self._graph_logged = False

        if self.enable_logging:
            logger.info(f"TensorBoard extended logging enabled (log_freq={log_freq})")

    def should_log(self, step: int) -> bool:
        """Check if logging should occur at this step.

        Args:
            step: Current training step

        Returns:
            True if should log at this step
        """
        return self.enable_logging and step % self.log_freq == 0

    # Model architecture

    def log_model_graph(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
    ) -> None:
        """Log model architecture graph to TensorBoard.

        Args:
            model: PyTorch model
            sample_input: Sample input tensor for tracing
        """
        if not self.enable_logging or self._graph_logged:
            return

        try:
            # Set model to eval mode for graph tracing
            was_training = model.training
            original_device = next(model.parameters()).device
            model.eval()

            # For graph logging, move everything to CPU to avoid device mismatch issues
            # This is more reliable than trying to sync all buffers/parameters to GPU
            model_cpu = model.cpu()
            input_cpu = sample_input.cpu()

            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                self.writer.add_graph(model_cpu, input_cpu)

            model.to(original_device)

            # Restore training mode
            if was_training:
                model.train()

            self._graph_logged = True
            logger.info("Model graph logged to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")

    # Histograms and distributions

    def log_weight_histograms(
        self,
        model: nn.Module,
        step: int,
        prefix: str = "weights",
    ) -> None:
        """Log model weight histograms to TensorBoard.

        Args:
            model: PyTorch model
            step: Current training step
            prefix: Tag prefix for the histograms
        """
        if not self.enable_logging:
            return

        for name, param in model.named_parameters():
            if param.requires_grad and param.data is not None:
                # Clean up name for better organization
                tag = f"{prefix}/{name.replace('.', '/')}"
                try:
                    self.writer.add_histogram(tag, param.data.cpu(), step)
                except Exception as e:
                    logger.debug(f"Failed to log histogram for {name}: {e}")

    def log_gradient_histograms(
        self,
        model: nn.Module,
        step: int,
        prefix: str = "gradients",
    ) -> None:
        """Log model gradient histograms to TensorBoard.

        Args:
            model: PyTorch model
            step: Current training step
            prefix: Tag prefix for the histograms
        """
        if not self.enable_logging:
            return

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                tag = f"{prefix}/{name.replace('.', '/')}"
                try:
                    self.writer.add_histogram(tag, param.grad.cpu(), step)
                except Exception as e:
                    logger.debug(f"Failed to log gradient histogram for {name}: {e}")

    def log_activation_histograms(
        self,
        activations: dict[str, torch.Tensor],
        step: int,
        prefix: str = "activations",
    ) -> None:
        """Log activation histograms to TensorBoard.

        Args:
            activations: Dictionary of layer name to activation tensor
            step: Current training step
            prefix: Tag prefix for the histograms
        """
        if not self.enable_logging:
            return

        for name, activation in activations.items():
            tag = f"{prefix}/{name}"
            try:
                self.writer.add_histogram(tag, activation.cpu(), step)
            except Exception as e:
                logger.debug(f"Failed to log activation histogram for {name}: {e}")

    # Scalars and time series

    def log_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        step: int,
        tag: str = "train/learning_rate",
    ) -> None:
        """Log current learning rate to TensorBoard.

        Args:
            optimizer: PyTorch optimizer
            step: Current training step
            tag: TensorBoard tag for the scalar
        """
        if not self.enable_logging:
            return

        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group["lr"]
            if len(optimizer.param_groups) > 1:
                self.writer.add_scalar(f"{tag}/group_{i}", lr, step)
            else:
                self.writer.add_scalar(tag, lr, step)

    def log_scalars(
        self,
        main_tag: str,
        scalars: dict[str, float],
        step: int,
    ) -> None:
        """Log multiple scalars to TensorBoard (grouped).

        Args:
            main_tag: Main tag for grouping
            scalars: Dictionary of scalar names to values
            step: Current training step
        """
        if not self.enable_logging:
            return

        self.writer.add_scalars(main_tag, scalars, step)

    def log_gradient_norm(
        self,
        model: nn.Module,
        step: int,
        tag: str = "train/gradient_norm",
    ) -> None:
        """Log gradient L2 norm to TensorBoard.

        Args:
            model: PyTorch model
            step: Current training step
            tag: TensorBoard tag for the scalar
        """
        if not self.enable_logging:
            return

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        self.writer.add_scalar(tag, total_norm, step)

    # Images and figures

    def log_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: list[str] | None,
        step: int,
        tag: str = "validation/confusion_matrix",
        normalize: bool = True,
    ) -> None:
        """Log confusion matrix as an image to TensorBoard.

        Args:
            confusion_matrix: NxN confusion matrix array
            class_names: List of class names
            step: Current training step
            tag: TensorBoard tag for the image
            normalize: Whether to normalize the matrix
        """
        if not self.enable_logging:
            return

        n_classes = confusion_matrix.shape[0]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]

        # Normalize if requested
        if normalize:
            cm = confusion_matrix.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, where=row_sums != 0)
        else:
            cm = confusion_matrix

        # Create figure
        fig, ax = plt.subplots(figsize=VISUALIZATION_DEFAULTS["CONFUSION_MATRIX_FIGSIZE"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

        plt.tight_layout()

        # Convert to tensor and log
        self.writer.add_figure(tag, fig, step)
        plt.close(fig)

    def log_per_class_metrics(
        self,
        metrics: Mapping[str, list[float]] | Iterable[tuple[str, list[float]]],
        class_names: list[str],
        step: int,
        tag_prefix: str = "per_class",
    ) -> None:
        """Log per-class metrics as bar chart.

        Args:
            metrics_dict: Dictionary mapping metric name to list of per-class values
            class_names: List of class names
            step: Current training step
            tag_prefix: Prefix for TensorBoard tags
        """
        if not self.enable_logging:
            return

        metrics_items = metrics.items() if isinstance(metrics, Mapping) else metrics
        for metric_name, values in metrics_items:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(class_names))
            ax.bar(x, values)
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_ylabel(metric_name)
            ax.set_title(f"Per-class {metric_name}")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            self.writer.add_figure(f"{tag_prefix}/{metric_name}", fig, step)
            plt.close(fig)

    def log_sample_predictions(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        step: int,
        tag: str = "predictions/samples",
        max_samples: int = 4,
    ) -> None:
        """Log sample prediction comparisons.

        Args:
            images: Input images [N, C, H, W]
            predictions: Predicted labels [N, H, W]
            targets: Ground truth labels [N, H, W]
            step: Current training step
            tag: TensorBoard tag
            max_samples: Maximum number of samples to show
        """
        if not self.enable_logging:
            return

        n_samples = min(max_samples, images.shape[0])

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Input image (first 3 channels as RGB)
            img = images[i, :3].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + VISUALIZATION_DEFAULTS["IMAGE_NORMALIZATION_EPSILON"])
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            # Prediction
            pred = predictions[i].cpu().numpy()
            axes[i, 1].imshow(pred, cmap="tab10")
            axes[i, 1].set_title("Prediction")
            axes[i, 1].axis("off")

            # Ground truth
            target = targets[i].cpu().numpy()
            axes[i, 2].imshow(target, cmap="tab10")
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis("off")

        plt.tight_layout()
        self.writer.add_figure(tag, fig, step)
        plt.close(fig)

    # Batch image logging

    def log_batch_images(
        self,
        images: torch.Tensor,
        step: int,
        tag: str = "input_samples/batch",
        max_samples: int = 8,
        normalize: bool = True,
    ) -> None:
        """Log a batch of input images to TensorBoard.

        Args:
            images: Input batch [N, C, H, W]
            step: Current training step
            tag: TensorBoard tag
            max_samples: Maximum number of samples to visualize
            normalize: Whether to normalize images to [0, 1]
        """
        if not self.enable_logging:
            return

        try:
            n_samples = min(max_samples, images.shape[0])
            batch = images[:n_samples]

            if normalize and batch.max() > 1.0:
                batch = (batch - batch.min()) / (batch.max() - batch.min() + VISUALIZATION_DEFAULTS["IMAGE_NORMALIZATION_EPSILON"])

            # Handle different numbers of channels
            if batch.shape[1] >= 3:
                # Use first 3 channels as RGB
                batch_rgb = batch[:, :3]
            elif batch.shape[1] == 1:
                # Repeat grayscale to 3 channels
                batch_rgb = batch.repeat(1, 3, 1, 1)
            else:
                # Pad to 3 channels
                batch_rgb = torch.nn.functional.pad(batch, (0, 0, 0, 0, 0, 3 - batch.shape[1]))

            self.writer.add_images(tag, batch_rgb, step, dataformats="NCHW")
            logger.info(f"Logged {n_samples} sample images at step {step}")

        except Exception as e:
            msg = TENSORBOARD_LOGGER_WARNINGS.HISTOGRAM_FAILED.format(name="batch_images", error=str(e))
            logger.warning(msg)

    # Distribution logging

    def log_activation_distributions(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        step: int,
        tag: str = "activations",
    ) -> None:
        """Log activation distributions through the model.

        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            step: Current training step
            tag: TensorBoard tag prefix
        """
        if not self.enable_logging:
            return

        try:
            activations = {}

            def hook(name: str) -> Any:
                def _hook(module: Any, input: Any, output: Any) -> None:
                    if isinstance(output, torch.Tensor):
                        activations[name] = output.detach()

                return _hook

            # Register hooks on conv and linear layers
            handles = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    handles.append(module.register_forward_hook(hook(name)))

            # Run forward pass
            with torch.no_grad():
                _ = model(sample_input)

            # Log activation statistics
            for name, activation in activations.items():
                if activation.numel() > 0:
                    # Flatten the activation
                    flat_act = activation.flatten()

                    # Log histogram
                    self.writer.add_histogram(f"{tag}/{name}", flat_act, step, bins=30)

                    # Log statistics as scalars
                    self.writer.add_scalar(f"{tag}/{name}/mean", float(flat_act.mean()), step)
                    self.writer.add_scalar(f"{tag}/{name}/std", float(flat_act.std()), step)
                    self.writer.add_scalar(f"{tag}/{name}/min", float(flat_act.min()), step)
                    self.writer.add_scalar(f"{tag}/{name}/max", float(flat_act.max()), step)

            # Remove hooks
            for handle in handles:
                handle.remove()

            logger.info(f"Logged activation distributions for {len(activations)} layers")

        except Exception as e:
            msg = TENSORBOARD_LOGGER_WARNINGS.HISTOGRAM_FAILED.format(name="activations", error=str(e))
            logger.warning(msg)

    def log_layer_outputs(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        step: int,
        layer_names: list[str] | None = None,
        tag: str = "layer_outputs",
    ) -> None:
        """Log output distributions from specific layers.

        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            step: Current training step
            layer_names: Specific layers to monitor (None = all conv layers)
            tag: TensorBoard tag prefix
        """
        if not self.enable_logging:
            return

        try:
            outputs = {}

            def hook(name: str) -> Any:
                def _hook(module: Any, input: Any, output: Any) -> None:
                    outputs[name] = output.detach() if isinstance(output, torch.Tensor) else output

                return _hook

            # Determine which layers to hook
            layers_to_hook = set()
            if layer_names:
                layers_to_hook = set(layer_names)
            else:
                # Hook all conv and linear layers by default
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        layers_to_hook.add(name)

            # Register hooks
            handles = []
            for name, module in model.named_modules():
                if name in layers_to_hook:
                    handles.append(module.register_forward_hook(hook(name)))

            # Run forward pass
            with torch.no_grad():
                _ = model(sample_input)

            # Log outputs
            for name, output in outputs.items():
                if isinstance(output, torch.Tensor) and output.numel() > 0:
                    flat_out = output.flatten()
                    self.writer.add_histogram(f"{tag}/{name}", flat_out, step, bins=30)

            # Remove hooks
            for handle in handles:
                handle.remove()

            logger.info(f"Logged outputs from {len(outputs)} layers")

        except Exception as e:
            msg = TENSORBOARD_LOGGER_WARNINGS.HISTOGRAM_FAILED.format(name="layer_outputs", error=str(e))
            logger.warning(msg)

    # Text logging

    def log_text(
        self,
        tag: str,
        text: str,
        step: int,
    ) -> None:
        """Log text to TensorBoard.

        Args:
            tag: TensorBoard tag
            text: Text content (supports markdown)
            step: Current training step
        """
        if not self.enable_logging:
            return

        self.writer.add_text(tag, text, step)

    def log_hyperparameters(
        self,
        hparams: Mapping[str, Any] | Iterable[tuple[str, Any]],
        metrics: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    ) -> None:
        """Log hyperparameters and associated metrics.

        Args:
            hparams: Dictionary of hyperparameter names to values
            metrics: Dictionary of metric names to final values
        """
        if not self.enable_logging:
            return

        hparam_items = hparams.items() if isinstance(hparams, Mapping) else hparams
        # Convert non-scalar hparams to strings
        clean_hparams = {}
        for k, v in hparam_items:
            if isinstance(v, (int, float, str, bool)):
                clean_hparams[k] = v
            elif v is None:
                clean_hparams[k] = "None"
            else:
                clean_hparams[k] = str(v)

        metrics_items: Iterable[tuple[str, float]] = []
        if metrics is not None:
            metrics_items = metrics.items() if isinstance(metrics, Mapping) else metrics
        metrics_map = dict(metrics_items)

        try:
            self.writer.add_hparams(clean_hparams, metrics_map)
            logger.info("Hyperparameters logged to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters: {e}")

    def log_config_summary(
        self,
        config: Mapping[str, Any] | Iterable[tuple[str, Any]],
        step: int = 0,
        tag: str = "config/summary",
    ) -> None:
        """Log configuration as formatted text.

        Args:
            config: Configuration dictionary
            step: Training step
            tag: TensorBoard tag
        """
        if not self.enable_logging:
            return

        text_lines = ["# Training Configuration\n"]
        config_items = config.items() if isinstance(config, Mapping) else config
        for key, value in sorted(config_items):
            text_lines.append(f"- **{key}**: `{value}`")

        self.log_text(tag, "\n".join(text_lines), step)

    # PR curves and metrics

    def log_pr_curve(
        self,
        tag: str,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        step: int,
        num_thresholds: int = 127,
    ) -> None:
        """Log precision-recall curve to TensorBoard.

        Args:
            tag: TensorBoard tag
            labels: Ground truth labels (binary)
            predictions: Prediction probabilities
            step: Current training step
            num_thresholds: Number of thresholds for the curve
        """
        if not self.enable_logging:
            return

        try:
            self.writer.add_pr_curve(
                tag,
                labels.cpu(),
                predictions.cpu(),
                step,
                num_thresholds=num_thresholds,
            )
        except Exception as e:
            logger.debug(f"Failed to log PR curve: {e}")

    # Embeddings

    def log_embeddings(
        self,
        embeddings: torch.Tensor,
        metadata: list[str] | None = None,
        label_img: torch.Tensor | None = None,
        step: int = 0,
        tag: str = "embeddings",
    ) -> None:
        """Log embeddings for visualization in TensorBoard projector.

        Args:
            embeddings: Embedding tensor [N, D]
            metadata: Labels for each embedding point
            label_img: Images for each embedding point [N, C, H, W]
            step: Current training step
            tag: TensorBoard tag
        """
        if not self.enable_logging:
            return

        try:
            self.writer.add_embedding(
                embeddings,
                metadata=metadata,
                label_img=label_img,
                global_step=step,
                tag=tag,
            )
            logger.info(f"Embeddings logged to TensorBoard ({embeddings.shape[0]} samples)")
        except Exception as e:
            logger.warning(f"Failed to log embeddings: {e}")

    # Custom figures

    def log_figure(
        self,
        tag: str,
        figure: plt.Figure,
        step: int,
        close: bool = True,
    ) -> None:
        """Log a matplotlib figure to TensorBoard.

        Args:
            tag: TensorBoard tag
            figure: Matplotlib figure
            step: Current training step
            close: Whether to close the figure after logging
        """
        if not self.enable_logging:
            return

        self.writer.add_figure(tag, figure, step, close=close)

    def log_loss_landscape(
        self,
        losses: np.ndarray,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        step: int,
        tag: str = "analysis/loss_landscape",
    ) -> None:
        """Log a 2D loss landscape visualization.

        Args:
            losses: 2D array of loss values
            x_range: (min, max) range for x-axis
            y_range: (min, max) range for y-axis
            step: Current training step
            tag: TensorBoard tag
        """
        if not self.enable_logging:
            return

        fig, ax = plt.subplots(figsize=VISUALIZATION_DEFAULTS["CONFUSION_MATRIX_FIGSIZE"])
        x = np.linspace(x_range[0], x_range[1], losses.shape[1])
        y = np.linspace(y_range[0], y_range[1], losses.shape[0])
        X, Y = np.meshgrid(x, y)

        contour = ax.contourf(X, Y, losses, levels=50, cmap="viridis")
        ax.figure.colorbar(contour, ax=ax, label="Loss")
        ax.set_xlabel("Direction 1")
        ax.set_ylabel("Direction 2")
        ax.set_title("Loss Landscape")

        plt.tight_layout()
        self.writer.add_figure(tag, fig, step)
        plt.close(fig)

    # Utility methods

    def flush(self) -> None:
        """Flush the TensorBoard writer."""
        if self.enable_logging:
            self.writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.enable_logging:
            self.writer.close()


def create_tensorboard_logger(
    writer: SummaryWriter,
    args: Any,
) -> TensorBoardLogger:
    """Create a TensorBoardLogger from parsed arguments.

    Args:
        writer: TensorBoard SummaryWriter instance
        args: Namespace containing configuration:
            - enable_tensorboard_logging: bool
            - tensorboard_log_freq: int

    Returns:
        Configured TensorBoardLogger instance
    """
    enable_logging = getattr(args, "enable_tensorboard_logging", TrainArgDefaults["enable_tensorboard_logging"])
    log_freq = getattr(args, "tensorboard_log_freq", TrainArgDefaults["tensorboard_log_freq"])

    return TensorBoardLogger(
        writer=writer,
        enable_logging=enable_logging,
        log_freq=log_freq,
    )
