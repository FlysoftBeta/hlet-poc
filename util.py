import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
max_value = 100

__all__ = [device, dtype, max_value]
