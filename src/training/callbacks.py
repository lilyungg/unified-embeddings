from pathlib import Path

import torch
from tqdm import tqdm


class TrainerCallback:
    """Optional side effects around the training loop."""

    def on_train_start(self, trainer):
        pass

    def on_metrics(self, trainer, split, metrics, step):
        pass

    def on_validation_end(self, trainer, epoch, metrics, is_best):
        pass

    def on_train_end(self, trainer):
        pass


class MetricsLoggerCallback(TrainerCallback):
    """Forward trainer metrics to an externally created logger."""

    def __init__(self, log_metrics):
        self.log_metrics = log_metrics

    def on_metrics(self, trainer, split, metrics, step):
        self.log_metrics(
            {f"{split}/{name}": value for name, value in metrics.items()},
            step=step,
        )


class ConsoleLoggerCallback(TrainerCallback):
    def on_metrics(self, trainer, split, metrics, step):
        values = " ".join(
            f"{name}={value:.4f}" for name, value in metrics.items()
        )
        tqdm.write(f"[{split}] step={step} {values}")


class ModelCheckpointCallback(TrainerCallback):
    """Persist and optionally restore the model selected by the trainer."""

    def __init__(self, path, restore_best=True):
        self.path = Path(path)
        self.restore_best = restore_best
        self.has_checkpoint = False

    def on_validation_end(self, trainer, epoch, metrics, is_best):
        if not is_best:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": trainer.model.state_dict(),
                "epoch": epoch,
                "global_step": trainer.global_step,
                "metrics": metrics,
            },
            self.path,
        )
        self.has_checkpoint = True

    def on_train_end(self, trainer):
        if not self.restore_best or not self.has_checkpoint:
            return

        checkpoint = torch.load(
            self.path,
            map_location="cpu",
            weights_only=True,
        )
        trainer.model.load_state_dict(checkpoint["model"])
