import random
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    print("Using CUDA")
else:
    print("Not using CUDA")