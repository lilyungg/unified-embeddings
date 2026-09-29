import time

import torch
from tqdm import tqdm

from src.metrics import MetricsContainer


class Trainer:

    def __init__(
        self,
        model,
        optimizers,
        compute_batch,
        evaluate_fn,
        train_dataloader,
        val_dataloader,
        num_steps,
        *,
        monitor,
        log_every_n_steps,
        monitor_mode="max",
        test_dataloader=None,
        schedulers=None,
        callbacks=None,
        patience=None,
        max_grad_norm=None,
    ):
        self.model = model
        self.optimizers = dict(optimizers)
        self.schedulers = dict(schedulers or {})
        self.compute_batch = compute_batch
        self.evaluate_fn = evaluate_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.callbacks = tuple(callbacks or ())

        self.num_steps = num_steps
        self.log_every_n_steps = log_every_n_steps
        self.max_grad_norm = max_grad_norm
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.patience = patience

        self.global_step = 0
        self.best_score = None
        self.best_epoch = None
        self.best_metrics = None

    def _call_callbacks(self, event, *args, **kwargs):
        for callback in self.callbacks:
            handler = getattr(callback, event, None)
            if handler is not None:
                handler(self, *args, **kwargs)

    def _log_metrics(self, split, metrics):
        self._call_callbacks(
            "on_metrics",
            split,
            metrics,
            self.global_step,
        )

    def _flush_train_window(self, scalars, steps, started):
        # Reading scalars waits for queued GPU work only at logging boundaries.
        metrics = scalars.get()
        elapsed = time.perf_counter() - started
        metrics["step_time_ms"] = elapsed * 1000 / steps

        # Runs use fixed-size batches with drop_last=True. For sequential data,
        # these are Dataset samples (users), not the number of valid targets.
        batch_size = self.train_dataloader.batch_size
        if batch_size is not None:
            metrics["samples_per_second"] = steps * batch_size / elapsed

        for name, optimizer in self.optimizers.items():
            for index, group in enumerate(optimizer.param_groups):
                key = f"lr/{name}"
                if len(optimizer.param_groups) > 1:
                    key += f"/group_{index}"
                lr = group["lr"]
                metrics[key] = lr.item() if isinstance(lr, torch.Tensor) else lr

        self._log_metrics("train", metrics)
        scalars.reset()
        return metrics

    def _train_epoch(self, epoch):
        self.model.train()
        window_scalars = MetricsContainer()
        epoch_scalars = MetricsContainer()
        window_steps = 0

        epoch_started = time.perf_counter()
        window_started = epoch_started
        progress = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
        )

        for batch_index, batch in enumerate(progress, start=1):
            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)

            result = self.compute_batch(self.model, batch)
            result["loss"].backward()
            if self.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm,
                )
                result = {**result, "grad_norm": grad_norm.detach()}

            for optimizer in self.optimizers.values():
                optimizer.step()
            for scheduler in self.schedulers.values():
                scheduler.step()

            window_scalars.update(result)
            epoch_scalars.update(result)
            window_steps += 1
            self.global_step += 1

            epoch_finished = batch_index == len(self.train_dataloader)
            num_steps_reached = self.global_step >= self.num_steps
            should_flush = (
                window_steps >= self.log_every_n_steps
                or epoch_finished
                or num_steps_reached
            )
            if not should_flush:
                continue

            metrics = self._flush_train_window(
                window_scalars,
                window_steps,
                window_started,
            )
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")
            window_steps = 0
            window_started = time.perf_counter()
            if num_steps_reached:
                break

        epoch_metrics = epoch_scalars.get()
        epoch_metrics["epoch_time_s"] = time.perf_counter() - epoch_started
        self._log_metrics("train_epoch", epoch_metrics)
        return epoch_metrics

    def _evaluate(self, dataloader, split):
        was_training = self.model.training
        self.model.eval()

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                metrics = self.evaluate_fn(self.model, dataloader)

            metrics = {
                name: value.item() if isinstance(value, torch.Tensor) else float(value)
                for name, value in metrics.items()
            }
        finally:
            self.model.train(was_training)

        metrics["evaluation_time_s"] = time.perf_counter() - started
        self._log_metrics(split, metrics)
        return metrics

    def _is_better(self, score):
        if self.best_score is None:
            return True
        if self.monitor_mode == "max":
            return score > self.best_score
        return score < self.best_score

    def evaluate(self, dataloader, split="test"):
        return self._evaluate(dataloader, split)

    def train(self):
        self._call_callbacks("on_train_start")
        epochs_without_improvement = 0

        epoch = 0
        while self.global_step < self.num_steps:
            epoch += 1
            self._train_epoch(epoch)
            val_metrics = self._evaluate(self.val_dataloader, "val")
            score = val_metrics[self.monitor]
            is_best = self._is_better(score)
            if is_best:
                self.best_score = score
                self.best_epoch = epoch
                self.best_metrics = val_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self._call_callbacks(
                "on_validation_end",
                epoch,
                val_metrics,
                is_best,
            )

            patience_reached = (
                self.patience is not None
                and epochs_without_improvement >= self.patience
            )
            if patience_reached:
                break

        self._call_callbacks("on_train_end")
        test_metrics = None
        if self.test_dataloader is not None:
            test_metrics = self._evaluate(self.test_dataloader, "test")

        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "test_metrics": test_metrics,
            "global_step": self.global_step,
        }
