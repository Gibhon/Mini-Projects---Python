import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

eps : float = 1e-7

def load_data(file_path : str) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter=";", skiprows=1)
    features_np = data[: , :-1]
    labels_np = data[:, -1]
    
    conditions = [
    (labels_np <= 5),
    (labels_np == 6),
    (labels_np >= 7)
    ]
    choices = [0, 1, 2]
    labels_np_collapsed = np.select(conditions, choices)
    
    features_np_final = features_np.astype(np.float32)
    labels_np_final = features_np.astype(np.float64)
    
    return features_np_final, labels_np_final

