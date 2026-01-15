"""Periodic Boundary Condition (PBC) utilities.

This module provides helper functions for handling periodic boundary conditions
in molecular simulations, including coordinate wrapping and minimum image convention.
"""

import numpy as np
import torch
from torch import Tensor


# =============================================================================
# NumPy implementations
# =============================================================================

def apply_minimum_image_np(delta_pos: np.ndarray, box_size: np.ndarray) -> np.ndarray:
    """Apply minimum image convention to displacement vectors (NumPy version).
    
    Args:
        delta_pos: Displacement vectors of shape (..., 3).
        box_size: Box dimensions of shape (3,).
        
    Returns:
        Wrapped displacement vectors in range [-L/2, L/2].
    """
    if box_size is None:
        return delta_pos
    box_size = np.asarray(box_size)
    return delta_pos - np.round(delta_pos / box_size) * box_size


def pbc_wrap_coords_np(coords: np.ndarray, box_size: np.ndarray) -> np.ndarray:
    """Wrap coordinates into periodic box (NumPy version).
    
    Args:
        coords: Coordinates of shape (..., 3).
        box_size: Box dimensions of shape (3,).
        
    Returns:
        Wrapped coordinates in range [0, L).
    """
    if box_size is None:
        return coords
    box_size = np.asarray(box_size)
    return coords - box_size * np.floor(coords / box_size)


def mean_pbc_np(pos1: np.ndarray, pos2: np.ndarray, box_size: np.ndarray = None) -> np.ndarray:
    """Calculate PBC-aware midpoint between two positions (NumPy version).
    
    Args:
        pos1: First position of shape (..., 3).
        pos2: Second position of shape (..., 3).
        box_size: Box dimensions of shape (3,), or None for non-periodic.
        
    Returns:
        Midpoint coordinates.
    """
    if box_size is not None:
        box_size = np.asarray(box_size)
        delta = pos2 - pos1
        delta = delta - np.round(delta / box_size) * box_size
        mean = pos1 + delta / 2
        mean = mean - np.floor(mean / box_size) * box_size
    else:
        mean = (pos1 + pos2) / 2
    return mean


# =============================================================================
# PyTorch implementations (JIT-compiled for performance)
# =============================================================================

@torch.jit.script
def apply_minimum_image(delta_pos: Tensor, box_size: Tensor) -> Tensor:
    """Apply minimum image convention to displacement vectors (JIT-compiled).
    
    Args:
        delta_pos: Displacement vectors of shape (..., 3).
        box_size: Box dimensions of shape (3,).
        
    Returns:
        Wrapped displacement vectors in range [-L/2, L/2].
    """
    return delta_pos - box_size * torch.round(delta_pos / box_size)


@torch.jit.script
def pbc_wrap_coords(coords: Tensor, box_size: Tensor) -> Tensor:
    """Wrap coordinates into periodic box (JIT-compiled).
    
    Args:
        coords: Coordinates of shape (..., 3).
        box_size: Box dimensions of shape (3,).
        
    Returns:
        Wrapped coordinates in range [0, L).
    """
    return coords - box_size * torch.floor(coords / box_size)


@torch.jit.script
def compute_pbc_distances(pos_i: Tensor, pos_j: Tensor, box_size: Tensor) -> Tensor:
    """Compute pairwise distances with PBC (JIT-compiled).
    
    Args:
        pos_i: First set of positions.
        pos_j: Second set of positions.
        box_size: Box dimensions.
        
    Returns:
        Distance tensor.
    """
    delta = pos_i - pos_j
    delta_pbc = apply_minimum_image(delta, box_size)
    return torch.norm(delta_pbc, dim=-1)


def apply_minimum_image_safe(delta_pos: Tensor, box_size: Tensor = None) -> Tensor:
    """Apply minimum image convention with None-safety.
    
    Args:
        delta_pos: Displacement vectors.
        box_size: Box dimensions, or None for non-periodic.
        
    Returns:
        Wrapped displacement vectors if box_size is provided, otherwise unchanged.
    """
    if box_size is None:
        return delta_pos
    if hasattr(delta_pos, 'device'):
        box_size = box_size.to(delta_pos.device)
    return apply_minimum_image(delta_pos, box_size)


def pbc_wrap_coords_safe(coords: Tensor, box_size: Tensor = None) -> Tensor:
    """Wrap coordinates with None-safety.
    
    Args:
        coords: Coordinates tensor.
        box_size: Box dimensions, or None for non-periodic.
        
    Returns:
        Wrapped coordinates if box_size is provided, otherwise unchanged.
    """
    if box_size is None:
        return coords
    if hasattr(coords, 'device'):
        box_size = box_size.to(coords.device)
    return pbc_wrap_coords(coords, box_size)
