import torch

# Check if CUDA device is present, if not, send a warning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
if device == "cpu":
    print("Warning: CUDA GPU is not present! The computations will be performed on CPU.")