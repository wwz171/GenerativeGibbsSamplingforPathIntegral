"""Gibbs sampler for path integral molecular dynamics.

This module implements a Gibbs sampler that uses learned flow models
to sample bead configurations in path integral simulations.
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Tuple, List
import numpy as np

from ..nn.visnet import radius_graph_pbc


class GibbsSampler:
    """Red-black Gibbs sampler for path integral beads.
    
    Performs alternating updates of odd and even indexed beads,
    sampling each from its conditional distribution given neighbors.
    
    Args:
        sampler: Flow sampler for generating new bead positions.
        box_size: Box dimensions for PBC, or None for non-periodic.
        device: Computation device.
        cutoff: Cutoff radius for graph construction.
        cache_update_interval: Steps between cache validity checks.
        cache_threshold: Maximum displacement before cache update.
        use_cache: Whether to cache graph structures.
    """

    def __init__(
        self,
        sampler,
        box_size: Optional[Tensor] = None,
        device: str = 'cpu',
        cutoff: Optional[float] = None,
        cache_update_interval: int = 5,
        cache_threshold: float = 2.0,
        use_cache: bool = True,
    ):
        self.sampler = sampler
        self.box_size = box_size
        self.device = torch.device(device)
        self.sampler = self.sampler.to(self.device)

        self.cutoff = cutoff if cutoff is not None else self.sampler.model.cutoff
        self.cutoff = float(self.cutoff)
        
        # Graph caching parameters
        self.cache_update_interval = cache_update_interval
        self.cache_threshold = cache_threshold
        self.use_cache = use_cache
        
        if self.box_size is None:
            self.use_cache = False
            print("Warning: box_size is None, disabling graph caching.")
        
        # Graph cache storage
        self.cached_pos: Dict[Tuple[int, int], Tensor] = {}
        self.cached_graph_dict: Dict[Tuple[int, int], Tensor] = {}
        self.cached_graph_lastupdate: Dict[Tuple[int, int], int] = {}
        
        self._init()
        self.update_times = 0
        self.update_stats = {'total_graphs': 0, 'updated_graphs': 0}
        
        print(
            f"GibbsSampler initialized on {self.device}, "
            f"cutoff={self.cutoff}, cache_interval={self.cache_update_interval}, "
            f"cache_threshold={self.cache_threshold}, use_cache={self.use_cache}"
        )

    def _init(self):
        """Validate initialization parameters."""
        if not callable(self.sampler):
            raise ValueError("sampler should be a callable function")
        
        if self.box_size is not None:
            if not isinstance(self.box_size, torch.Tensor):
                self.box_size = torch.tensor(self.box_size, dtype=torch.float32)
            if self.box_size.ndim != 1 or self.box_size.shape[0] != 3:
                raise ValueError("box_size should be a 1D tensor of shape (3,)")
            self.box_size = self.box_size.to(self.device)
    
    def _update_graph_edge(self, pos: Tensor) -> Tensor:
        """Compute edge indices for a single configuration."""
        return radius_graph_pbc(pos, self.cutoff, box_size=self.box_size)
    
    def _update_graph_cache_batch(
        self,
        current_pos: Tensor,
        update_idx: List[Tuple[int, int]]
    ):
        """Update graph cache for specified indices."""
        for i, j in update_idx:
            pos_ij = current_pos[i, j]
            key_ij = (i, j)
            edge_idx_ij = self._update_graph_edge(pos_ij)
            self.cached_graph_dict[key_ij] = edge_idx_ij
            self.cached_graph_lastupdate[key_ij] = self.update_times
            self.cached_pos[key_ij] = pos_ij.clone()
    
    def _update_graph_cache(
        self,
        current_pos: Tensor,
        update_idx: Optional[List[Tuple[int, int]]] = None
    ):
        """Update graph cache for all or specified configurations."""
        if not self.use_cache:
            return

        if update_idx is None:
            update_idx = [
                (i, j) 
                for i in range(current_pos.shape[0]) 
                for j in range(current_pos.shape[1])
            ]
        
        self._update_graph_cache_batch(current_pos, update_idx)
            
    def _should_update_cache(
        self,
        current_pos: Tensor
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check which graph caches need updating.
        
        Returns:
            Tuple of (needs_update, list of indices to update).
        """
        T, P = current_pos.shape[:2]

        if not self.use_cache:
            return False, []
        
        if not self.cached_pos:
            return True, [(i, j) for i in range(T) for j in range(P)]
        
        update_idx = []
        current_time = self.update_times
        
        keys_to_check = []
        positions_to_check = []
        indices_to_check = []
        
        for i in range(T):
            for j in range(P):
                key_ij = (i, j)
                
                if key_ij not in self.cached_pos:
                    update_idx.append((i, j))
                    continue
                
                time_diff = current_time - self.cached_graph_lastupdate.get(
                    key_ij, -self.cache_update_interval - 1
                )
                if time_diff >= self.cache_update_interval:
                    update_idx.append((i, j))
                    continue

                keys_to_check.append(key_ij)
                positions_to_check.append(current_pos[i, j])
                indices_to_check.append((i, j))

        # Batch displacement checking
        if keys_to_check:
            current_positions = torch.stack(positions_to_check)
            cached_positions = torch.stack(
                [self.cached_pos[k] for k in keys_to_check]
            )

            if self.box_size is not None:
                diff = current_positions - cached_positions
                diff = diff - self.box_size * torch.round(diff / self.box_size)
            else:
                diff = current_positions - cached_positions

            max_displacements = diff.norm(dim=-1).amax(dim=-1)
            mask = max_displacements >= self.cache_threshold
            
            if mask.any():
                idx_gpu = mask.nonzero(as_tuple=True)[0]
                for k in idx_gpu.tolist():
                    update_idx.append(indices_to_check[k])

        total_graphs = T * P
        self.update_stats['total_graphs'] = total_graphs
        self.update_stats['updated_graphs'] = len(update_idx)
        
        return len(update_idx) > 0, update_idx

    def _pbc_wrap_coords(self, coords: Tensor) -> Tensor:
        """Wrap coordinates into periodic box."""
        if self.box_size is None:
            return coords
        
        coords = coords.to(self.device)
        box = self.box_size.view((1,) * (coords.ndim - 1) + (-1,))
        return coords - box * torch.round(coords / box)

    def _pbc_distance(self, pos1: Tensor, pos2: Tensor) -> Tensor:
        """Calculate PBC-aware distance."""
        if self.box_size is None:
            diff = pos1 - pos2
        else:
            diff = pos1 - pos2
            diff = diff - self.box_size * torch.round(diff / self.box_size)
        return torch.norm(diff, dim=-1)

    def _mean_pbc(self, p1: Tensor, p2: Tensor) -> Tensor:
        """Calculate PBC-aware midpoint."""
        if self.box_size is None:
            return (p1 + p2) / 2
        
        delta = self._pbc_wrap_coords(p2 - p1)
        return self._pbc_wrap_coords(p1 + delta / 2)
    
    def _get_edge_indices_batch(
        self,
        data_shape: Tuple[int, ...],
        bead_indices: List[int]
    ) -> List[Tensor]:
        """Get cached edge indices for batch."""
        T = data_shape[0]
        updated_keys = [(i, j) for i in range(T) for j in bead_indices]
        return [self.cached_graph_dict[key] for key in updated_keys]

    @torch.no_grad()
    def update(self, verbose: bool = False):
        """Perform one step of red-black Gibbs sampling.
        
        Updates odd-indexed beads first, then even-indexed beads.
        Each bead is sampled from its conditional distribution given
        its neighbors (previous and next beads in the ring polymer).
        """
        data = self.current_data.to(self.device)
        num_beads = data.shape[1]

        if num_beads <= 1:
            return

        # Check and update graph cache
        should_update, update_idx = self._should_update_cache(data)
        if should_update:
            self._update_graph_cache(data, update_idx)

        # Generate odd and even indices
        odd_idx = torch.arange(1, num_beads, 2, device=self.device)
        even_idx = torch.arange(0, num_beads, 2, device=self.device)
        odd_idx_cpu = torch.arange(1, num_beads, 2)
        even_idx_cpu = torch.arange(0, num_beads, 2)

        if self.use_cache:
            edge_indices_odd = self._get_edge_indices_batch(
                data.shape, odd_idx_cpu.tolist()
            )
            edge_indices_even = self._get_edge_indices_batch(
                data.shape, even_idx_cpu.tolist()
            )
        else:
            edge_indices_odd = None
            edge_indices_even = None

        odd_idx = odd_idx_cpu.to(self.device, non_blocking=True)
        even_idx = even_idx_cpu.to(self.device, non_blocking=True)

        # Update odd-indexed beads
        if len(odd_idx) > 0:
            left_idx = odd_idx - 1
            right_idx = (odd_idx + 1) % num_beads
            left_points = data[:, left_idx]
            right_points = data[:, right_idx]
            
            centroid_points = self._mean_pbc(left_points, right_points)
            current_shape = centroid_points.shape
            reshaped_points = centroid_points.view(-1, current_shape[-2], current_shape[-1])
            
            updated_points = self.sampler(
                reshaped_points,
                edge_index=edge_indices_odd,
                fix_graph=True,
            )
            
            updated_points = updated_points.view(current_shape)
            updated_points = self._pbc_wrap_coords(updated_points)
            data[:, odd_idx] = updated_points

        # Update even-indexed beads
        if len(even_idx) > 0:
            left_idx = (even_idx - 1) % num_beads
            right_idx = (even_idx + 1) % num_beads
            left_points = data[:, left_idx]
            right_points = data[:, right_idx]

            centroid_points = self._mean_pbc(left_points, right_points)
            current_shape = centroid_points.shape
            reshaped_points = centroid_points.view(-1, current_shape[-2], current_shape[-1])

            updated_points = self.sampler(
                reshaped_points,
                edge_index=edge_indices_even,
                fix_graph=True,
            )
            updated_points = updated_points.view(current_shape)
            updated_points = self._pbc_wrap_coords(updated_points)
            data[:, even_idx] = updated_points
        
        self.current_data = data
        self.update_times += 1

    def cal_rg(
        self,
        data: Optional[Tensor] = None,
        box_size: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate radius of gyration with PBC alignment.
        
        Args:
            data: Bead configurations (B, num_beads, num_atoms, 3).
            box_size: Box dimensions for PBC.
            
        Returns:
            Mean radius of gyration.
        """
        if data is None:
            data = self.current_data
        if box_size is None:
            box_size = self.box_size

        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        if box_size is not None:
            box_size = torch.as_tensor(
                box_size, dtype=data.dtype, device=data.device
            )
        
        if data.ndim != 4:
            raise ValueError("data should be of shape (B, num_beads, num_atoms, dim)")

        if box_size is not None:
            ref = data[:, 0, :, :]
            diff = data - ref.unsqueeze(1)
            diff = diff - torch.round(diff / box_size) * box_size
            aligned_data = ref.unsqueeze(1) + diff
        else:
            aligned_data = data

        data_mean = torch.mean(aligned_data, dim=1, keepdim=True)
        diff = aligned_data - data_mean
        rg = torch.mean(torch.sum(diff**2, dim=-1))
        return rg

    def sample(
        self,
        init_data: Tensor,
        num_beads: int,
        max_iterations: int,
        record_freq: int = 10,
        early_stop_condition: Optional[callable] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List]:
        """Perform Gibbs sampling from initial configuration.
        
        Args:
            init_data: Initial configuration (B, [num_beads], num_atoms, 3).
            num_beads: Target number of beads.
            max_iterations: Maximum number of Gibbs updates.
            record_freq: Frequency for recording history.
            early_stop_condition: Optional stopping criterion.
            verbose: Whether to print progress.
            
        Returns:
            Tuple of (final_data, history) where history contains
            (step, data, num_beads, rg) tuples.
        """
        # Process initial data
        if init_data.ndim == 3:
            init_data = init_data.unsqueeze(1)
        elif init_data.ndim != 4:
            raise ValueError(
                "init_data should be of shape (B, num_beads, num_atoms, dim)"
            )

        self.current_data = init_data.to(self.device)
        self.update_times = 0
        
        # Reset cache
        self.cached_pos.clear()
        self.cached_graph_dict.clear()
        self.cached_graph_lastupdate.clear()

        # Expand to target number of beads
        current_beads = self.current_data.shape[1]
        if current_beads < num_beads:
            repeat_factor = (num_beads + current_beads - 1) // current_beads
            expanded_data = self.current_data.repeat(1, repeat_factor, 1, 1)
            self.current_data = expanded_data[:, :num_beads]
        
        # Initialize graph cache
        self._update_graph_cache(self.current_data)

        initial_rg = self.cal_rg()
        print(f"Initial Rg: {initial_rg.item()}, num_beads: {self.current_data.shape[1]}")

        history = []
        
        # Sampling loop
        for step in range(max_iterations):
            self.update(verbose=verbose)
            
            if self.update_times % record_freq == 0:
                current_rg = self.cal_rg()
                saved_data = self.current_data.clone()
                history.append((
                    self.update_times,
                    saved_data.detach().cpu().numpy(),
                    self.current_data.shape[1],
                    current_rg.item(),
                ))
                if verbose:
                    print(f"Step: {self.update_times}/{max_iterations}, Rg: {current_rg.item()}")

            if early_stop_condition is not None:
                if early_stop_condition(self.current_data):
                    print(f"Early stopping at step {self.update_times}")
                    break
        
        # Final record
        final_rg = self.cal_rg()
        saved_data = self.current_data.clone()
        history.append((
            self.update_times,
            saved_data.detach().cpu().numpy(),
            self.current_data.shape[1],
            final_rg.item(),
        ))
        
        return saved_data.cpu().numpy(), history
