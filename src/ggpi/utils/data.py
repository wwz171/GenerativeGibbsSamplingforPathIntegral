"""Data processing utilities for molecular datasets.

This module provides dataset classes and helper functions for preparing
training data from molecular dynamics trajectories.
"""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple

from .pbc import mean_pbc_np


class MolecularDataset(Dataset):
    """Dataset for molecular configurations with target and conditional positions.
    
    Args:
        pos_target: Target positions of shape (B, N, 3).
        pos_cond: Conditional positions of shape (B, N, 3).
        Z: Atomic numbers of shape (N,) or (B, N).
    """
    
    def __init__(self, pos_target: Tensor, pos_cond: Tensor, Z: Tensor):
        pos_target = torch.as_tensor(pos_target, dtype=torch.float32)
        pos_cond = torch.as_tensor(pos_cond, dtype=torch.float32)
        Z = torch.as_tensor(Z, dtype=torch.long)
        
        if pos_target.ndim != 3 or pos_cond.ndim != 3:
            raise ValueError("pos_target and pos_cond must be (B, N, 3) tensors")

        if Z.ndim == 1:
            Z = Z.unsqueeze(0).expand(pos_target.shape[0], -1)
            
        if pos_target.shape != pos_cond.shape:
            raise ValueError("pos_target and pos_cond must have the same shape")

        self.num_atoms = pos_target.shape[1]
        self.pos_target = pos_target
        self.pos_cond = pos_cond
        self.Z = Z

    def __len__(self) -> int:
        return self.pos_target.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.pos_target[idx], self.pos_cond[idx], self.Z[idx]


def get_training_tuples(
    data: np.ndarray,
    box_size: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training tuples from path integral bead configurations.
    
    For each bead, creates a (target, condition) pair where:
    - target: the bead position
    - condition: midpoint of neighboring beads (PBC-aware if box_size provided)
    
    Args:
        data: Bead configurations of shape (B, num_beads, num_atoms, 3).
        box_size: Box dimensions for PBC, or None for non-periodic.
        
    Returns:
        Tuple of (targets, conditions) each of shape (B*num_beads, num_atoms, 3).
    """
    if box_size is not None:
        box_size = np.asarray(box_size)
        
    if data.ndim != 4:
        raise ValueError("Data should be of shape (B, num_beads, num_atoms, dim)")
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Get neighboring beads (circular)
    data_left = np.roll(data, shift=1, axis=1)
    data_right = np.roll(data, shift=-1, axis=1)
    
    # Target is the original bead position
    data_target = data
    
    # Condition is the midpoint of neighbors
    if box_size is not None:
        data_cond = mean_pbc_np(data_left, data_right, box_size)
    else:
        data_cond = (data_left + data_right) / 2

    # Flatten batch and bead dimensions
    data_target = data_target.reshape(-1, data_target.shape[-2], data_target.shape[-1])
    data_cond = data_cond.reshape(-1, data_cond.shape[-2], data_cond.shape[-1])
    
    return data_target, data_cond
