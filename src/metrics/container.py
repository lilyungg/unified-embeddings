import torch


class MetricsContainer:
    def __init__(self):
        self.totals = {}
        self.weights = {}

    def update(self, metrics, weight=1):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().reshape(())
            else:
                value = torch.tensor(float(value), dtype=torch.float32)

            if key not in self.totals:
                self.totals[key] = value * weight
                self.weights[key] = weight
                continue

            if value.device != self.totals[key].device:
                value = value.to(self.totals[key].device)
            self.totals[key] += value * weight
            self.weights[key] += weight

    def get(self, names=None):
        names = names or list(self.totals)
        return {
            key: (self.totals[key] / self.weights[key]).item()
            for key in names
            if key in self.totals
        }

    def reset(self, names=None):
        if names is None:
            self.totals.clear()
            self.weights.clear()
            return

        for key in names:
            self.totals.pop(key, None)
            self.weights.pop(key, None)
