from torch.utils.data import DataLoader


class _DeviceDataLoader:
    def __init__(self, dataloader, device, non_blocking, batch_processor):
        self.dataloader = dataloader
        self.device = device
        self.non_blocking = non_blocking
        self.batch_processor = batch_processor

    def __iter__(self):
        for batch in self.dataloader:
            if self.device is not None:
                batch = {
                    key: value.to(self.device, non_blocking=self.non_blocking)
                    for key, value in batch.items()
                }
            if self.batch_processor is not None:
                batch = self.batch_processor(batch)
            yield batch

    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        return getattr(self.dataloader, name)


def create_dataloader(
    dataset,
    batch_size=1,
    *,
    collate_fn=None,
    device="cuda",
    batch_processor=None,
    num_workers=4,
    pin_memory=None,
    non_blocking=None,
    persistent_workers=None,
    **kwargs,
):
    """Yield flat tensor dicts: collate, transfer, then optionally process.

    Transfer and processing run in the consuming process, not loader workers.
    device=None skips transfer.
    """
    use_cuda = device == "cuda" or (
        device is not None and device.startswith("cuda:")
    )

    if pin_memory is None:
        pin_memory = use_cuda
    if non_blocking is None:
        non_blocking = use_cuda
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    inner_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **kwargs,
    )

    return _DeviceDataLoader(
        inner_dataloader,
        device,
        non_blocking,
        batch_processor,
    )
