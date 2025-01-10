import torch
import gymnasium
import pettingzoo

print(f"PyTorch version: {torch.__version__}")
print(f"Gymnasium version: {gymnasium.__version__}")
print(f"PettingZoo version: {pettingzoo.__version__}")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
