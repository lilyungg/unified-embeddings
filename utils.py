import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_config(config_file: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location('config_module', config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config
