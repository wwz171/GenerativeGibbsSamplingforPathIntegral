"""Conditional Flow Matching for molecular sampling.

This module provides classes for training and sampling with conditional
flow matching models, specifically designed for path integral molecular dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from typing import Optional, List

from ..utils.pbc import apply_minimum_image, pbc_wrap_coords


def weighted_mse_loss(
    pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None
) -> Tensor:
    """Compute weighted MSE loss.
    
    Args:
        pred: Predictions of shape (B, N, D).
        target: Targets of shape (B, N, D).
        weight: Optional weights of shape (B,).
        
    Returns:
        Scalar loss value.
    """
    if weight is None:
        return F.mse_loss(pred, target, reduction='mean')
    else:
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        loss = F.mse_loss(pred, target, reduction='none')
        weighted_loss = loss * weight
        total_elements = weight.sum() * pred.shape[1] * pred.shape[2]
        return weighted_loss.sum() / total_elements


class ConditionalLinearFlowMatching(nn.Module):
    """Conditional linear flow matching trainer.
    
    Implements the conditional flow matching objective for training
    velocity prediction networks.
    
    Args:
        model: Velocity prediction network.
        prior_sigma: Standard deviation of the prior distribution.
        extra_sigma: Additional noise for regularization.
        box_size: Box dimensions for PBC, or None for non-periodic.
    """
    
    def __init__(
        self,
        model: nn.Module,
        prior_sigma: Tensor,
        extra_sigma: float = 0,
        box_size: Optional[Tensor] = None,
    ):
        super().__init__()

        self.model = model
        if prior_sigma.ndim == 1:
            prior_sigma = prior_sigma.unsqueeze(0).unsqueeze(-1).to(model.device)
        self.prior_sigma = prior_sigma
        self.extra_sigma = extra_sigma
        self.box_size = box_size
        print(f"Extra noise sigma: {self.extra_sigma}")

    def forward(
        self,
        pos_target: Tensor,
        pos_cond: Tensor,
        h_target: Tensor,
        t_batch: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute flow matching loss.
        
        Args:
            pos_target: Target positions (B, N, D).
            pos_cond: Conditional positions (B, N, D).
            h_target: Node features (B, N) or (B, N, F).
            t_batch: Optional time values (B,). Random if None.
            edge_index: Optional edge indices.
            edge_attr: Optional edge attributes.
            weight: Optional sample weights (B,).
            
        Returns:
            Scalar loss value.
        """
        B, N, D = pos_target.shape
        device = pos_target.device
        
        # Sample time uniformly
        if t_batch is None:
            t = torch.rand(B, device=device)
        else:
            t = t_batch
        
        # Linear interpolation from prior
        x_0 = torch.randn_like(pos_target) * self.prior_sigma + pos_cond    
        eps = torch.randn_like(pos_target) * self.extra_sigma

        # Target velocity
        v_target = pos_target - x_0
        if self.box_size is not None:
            v_target = apply_minimum_image(v_target, self.box_size)

        # Interpolated position
        x_t = x_0 + t.unsqueeze(-1).unsqueeze(-1) * v_target + eps
        
        # Input velocity (relative to condition)
        v_init = x_t - pos_cond
        if self.box_size is not None:
            v_init = apply_minimum_image(v_init, self.box_size)

        # Predict velocity
        _, v_pred, _ = self.model(
            x_t, h_target, v_init, t,
            box_size=self.box_size,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Compute loss
        residual_target = v_target
        if self.box_size is not None:
            residual_target = apply_minimum_image(residual_target, self.box_size)
            
        if weight is None:
            loss = F.mse_loss(v_pred, residual_target, reduction='mean')
        else:
            loss = weighted_mse_loss(v_pred, residual_target, weight=weight)
            
        return loss


class ConditionalFlowSampler(nn.Module):
    """Sampler for conditional flow models.
    
    Generates samples by integrating the learned velocity field
    from time 0 to 1 using ODE solvers.
    
    Args:
        model: Trained velocity prediction network.
        Z: Atomic numbers of shape (N,).
        prior_sigma: Standard deviation of the prior.
        box_size: Box dimensions for PBC.
        sample_method: Integration method ('euler' or 'heun').
        num_steps: Number of integration steps.
        device: Computation device.
        fix_graph: Whether to reuse graph structure.
        max_batch_size: Maximum batch size for memory efficiency.
        verbose: Whether to show progress bars.
        return_cpu: Whether to return results on CPU.
    """
    
    def __init__(
        self,
        model: nn.Module,
        Z: Tensor,
        prior_sigma: Tensor,
        box_size: Optional[Tensor] = None,
        sample_method: str = 'heun',
        num_steps: int = 10,
        device: str = 'cpu',
        fix_graph: bool = True,
        max_batch_size: int = 64,
        verbose: bool = False,
        return_cpu: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        self.model = model
        self.device = torch.device(device)
        
        # Process atomic numbers
        Z = torch.as_tensor(Z, dtype=torch.long, device=self.device)
        Z = Z.unsqueeze(0).unsqueeze(-1)
        self.Z = Z.to(self.device, dtype=torch.long)
        
        # Process prior sigma
        prior_sigma = torch.as_tensor(
            prior_sigma, dtype=torch.float32, device=self.device
        )
        if prior_sigma.ndim == 1:
            prior_sigma = prior_sigma.unsqueeze(0).unsqueeze(-1)
        self.prior_sigma = prior_sigma
        
        # Box size for PBC
        self.box_size = box_size.to(self.device) if box_size is not None else None
        
        # Sampling configuration
        if sample_method not in ['euler', 'heun']:
            raise ValueError(
                f"Invalid sample_method: {sample_method}. "
                "Must be 'euler' or 'heun'."
            )
        self.sample_method = getattr(self, f'sample_{sample_method}', None)
        self.num_steps = num_steps
        
        # Performance settings
        self.fix_graph = fix_graph
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        self.return_cpu = return_cpu

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
        
        print(
            f"ConditionalFlowSampler initialized: "
            f"method={sample_method}, steps={num_steps}, "
            f"fix_graph={fix_graph}"
        )

    def merge_graphs(
        self,
        edge_indices: List[Tensor],
        num_nodes_per_graph: int
    ) -> Tensor:
        """Merge multiple graphs into a batched graph.
        
        Args:
            edge_indices: List of edge index tensors.
            num_nodes_per_graph: Number of nodes in each graph.
            
        Returns:
            Merged edge index tensor.
        """
        if not isinstance(edge_indices, list) or len(edge_indices) == 0:
            return edge_indices
        
        device = edge_indices[0].device
        merged_edges = []
        node_offset = 0
        
        for edge_idx in edge_indices:
            if edge_idx.shape[1] > 0:
                offset_edge = edge_idx + node_offset
                merged_edges.append(offset_edge)
            node_offset += num_nodes_per_graph
        
        if merged_edges:
            return torch.cat(merged_edges, dim=1)
        return torch.empty((2, 0), dtype=torch.long, device=device)

    @torch.no_grad()
    def sample_euler(
        self,
        pos_cond: Tensor,
        z: Tensor,
        num_steps: int = 100,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        fix_graph: bool = False,
        verbose: bool = False,
    ) -> Tensor:
        """Sample using Euler integration.
        
        Args:
            pos_cond: Conditional positions (B, N, D).
            z: Atomic numbers (B, N, 1).
            num_steps: Number of integration steps.
            edge_index: Optional pre-computed edges.
            edge_attr: Optional edge attributes.
            fix_graph: Whether to reuse graph structure.
            verbose: Whether to show progress.
            
        Returns:
            Generated positions (B, N, D).
        """
        B, N, D = pos_cond.shape
        device = pos_cond.device

        if edge_index is not None:
            fix_graph = True
            if isinstance(edge_index, list):
                edge_index = self.merge_graphs(edge_index, N)

        x_t = torch.randn_like(pos_cond) * self.prior_sigma + pos_cond
        if num_steps == 0:
            return x_t
        
        dt = 1.0 / num_steps
        iterator = tqdm(range(num_steps), desc="Sampling") if verbose else range(num_steps)

        for step in iterator:
            t = torch.full((B,), step * dt, device=device)
            v_init = x_t - pos_cond
            if self.box_size is not None:
                v_init = apply_minimum_image(v_init, self.box_size)
                
            if fix_graph:
                _, v_t, edge_index = self.model(
                    x_t, z, v_init, t=t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    box_size=self.box_size,
                )
            else:
                _, v_t, _ = self.model(
                    x_t, z, v_init, t=t, box_size=self.box_size
                )

            x_t = x_t + v_t * dt
        
        if self.box_size is not None:
            x_t = pbc_wrap_coords(x_t, self.box_size)
        return x_t

    @torch.no_grad()
    def sample_heun(
        self,
        pos_cond: Tensor,
        z: Tensor,
        num_steps: int = 50,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        fix_graph: bool = False,
        verbose: bool = False,
    ) -> Tensor:
        """Sample using Heun's method (second-order Runge-Kutta).
        
        Args:
            pos_cond: Conditional positions (B, N, D).
            z: Atomic numbers (B, N, 1).
            num_steps: Number of integration steps.
            edge_index: Optional pre-computed edges.
            edge_attr: Optional edge attributes.
            fix_graph: Whether to reuse graph structure.
            verbose: Whether to show progress.
            
        Returns:
            Generated positions (B, N, D).
        """
        B, N, D = pos_cond.shape
        device = pos_cond.device

        if edge_index is not None:
            fix_graph = True
            if isinstance(edge_index, list):
                edge_index = self.merge_graphs(edge_index, N)
        
        x_t = torch.randn_like(pos_cond) * self.prior_sigma + pos_cond
        if num_steps == 0:
            return x_t
        
        dt = 1.0 / num_steps
        iterator = tqdm(range(num_steps), desc="Heun Sampling") if verbose else range(num_steps)
        
        for step in iterator:
            t = torch.full((B,), step * dt, device=device)
            
            # Predictor step
            v_init_1 = x_t - pos_cond
            if self.box_size is not None:
                v_init_1 = apply_minimum_image(v_init_1, self.box_size)
            
            if fix_graph:
                _, v1, edge_index = self.model(
                    x_t, z, v_init_1, t=t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    box_size=self.box_size,
                )
            else:
                _, v1, _ = self.model(
                    x_t, z, v_init_1, t=t, box_size=self.box_size
                )

            # Predicted position
            x_pred = x_t + v1 * dt
            if self.box_size is not None:
                x_pred = pbc_wrap_coords(x_pred, self.box_size)
            
            # Corrector step
            t_next = torch.full((B,), (step + 1) * dt, device=device)
            
            v_init_2 = x_pred - pos_cond
            if self.box_size is not None:
                v_init_2 = apply_minimum_image(v_init_2, self.box_size)
            
            if fix_graph:
                _, v2, _ = self.model(
                    x_pred, z, v_init_2, t=t_next,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    box_size=self.box_size,
                )
            else:
                _, v2, _ = self.model(
                    x_pred, z, v_init_2, t=t_next, box_size=self.box_size
                )
            
            # Average velocity
            x_t = x_t + 0.5 * (v1 + v2) * dt
            
            if self.box_size is not None:
                x_t = pbc_wrap_coords(x_t, self.box_size)
        
        return x_t

    @torch.no_grad()
    def sample_batches(
        self,
        pos_cond: Tensor,
        Z_cond: Tensor,
        num_steps: int = 10,
        sample_method: str = 'heun',
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        fix_graph: bool = False,
        verbose: bool = False,
    ) -> Tensor:
        """Process input in batches for memory efficiency.
        
        Args:
            pos_cond: Conditional positions (B, N, D).
            Z_cond: Atomic numbers (B, N, 1).
            num_steps: Number of integration steps.
            sample_method: Integration method.
            edge_index: Optional edge indices.
            edge_attr: Optional edge attributes.
            fix_graph: Whether to reuse graph.
            verbose: Whether to show progress.
            
        Returns:
            Generated positions (B, N, D).
        """
        if sample_method not in ['euler', 'heun']:
            raise ValueError(
                f"Invalid sample_method: {sample_method}. "
                "Must be 'euler' or 'heun'."
            )
        
        if pos_cond.ndim != 3 or Z_cond.ndim != 3:
            raise ValueError(
                f"pos_cond and Z_cond must be 3D tensors, "
                f"got {pos_cond.ndim}D and {Z_cond.ndim}D"
            )
        if pos_cond.shape[0] != Z_cond.shape[0]:
            raise ValueError(
                f"Batch size mismatch: {pos_cond.shape[0]} vs {Z_cond.shape[0]}"
            )

        pos_cond_batches = torch.split(pos_cond, self.max_batch_size, dim=0)
        Z_cond_batches = torch.split(Z_cond, self.max_batch_size, dim=0)
        results = []
        
        if edge_index is None:
            edge_index_batches = [None] * len(pos_cond_batches)
        elif isinstance(edge_index, list):
            edge_index_batches = [
                edge_index[i:i + self.max_batch_size]
                for i in range(0, len(edge_index), self.max_batch_size)
            ]
        else:
            edge_index_batches = [edge_index] * len(pos_cond_batches)

        sample_method_func = getattr(self, f'sample_{sample_method}', None)

        for i in range(len(pos_cond_batches)):
            temp_results = sample_method_func(
                pos_cond_batches[i],
                Z_cond_batches[i],
                num_steps=num_steps,
                edge_index=edge_index_batches[i],
                edge_attr=edge_attr,
                fix_graph=fix_graph,
                verbose=verbose,
            )
            results.append(temp_results)
        
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def forward(
        self,
        pos_cond: Tensor,
        sample_method: Optional[str] = None,
        num_steps: Optional[int] = None,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        fix_graph: Optional[bool] = None,
        verbose: Optional[bool] = None,
        return_cpu: Optional[bool] = None,
    ) -> Tensor:
        """Generate samples from conditional positions.
        
        Args:
            pos_cond: Conditional positions.
            sample_method: Integration method (uses default if None).
            num_steps: Integration steps (uses default if None).
            edge_index: Optional edge indices.
            edge_attr: Optional edge attributes.
            fix_graph: Whether to reuse graph (uses default if None).
            verbose: Whether to show progress (uses default if None).
            return_cpu: Whether to return on CPU (uses default if None).
            
        Returns:
            Generated positions.
        """
        # Use defaults
        sample_method = sample_method or self.sample_method.__name__.replace('sample_', '')
        num_steps = num_steps or self.num_steps
        fix_graph = fix_graph if fix_graph is not None else self.fix_graph
        verbose = verbose if verbose is not None else self.verbose
        return_cpu = return_cpu if return_cpu is not None else self.return_cpu
        
        if not hasattr(self, 'Z') or self.Z is None:
            raise ValueError("Z tensor must be properly initialized.")
        
        Z_cond = self.Z.expand(pos_cond.shape[0], -1, -1).to(self.device)
        pos_cond = torch.as_tensor(pos_cond, device=self.device, dtype=torch.float32)
        
        results = self.sample_batches(
            pos_cond, Z_cond,
            num_steps=num_steps,
            sample_method=sample_method,
            edge_index=edge_index,
            edge_attr=edge_attr,
            fix_graph=fix_graph,
            verbose=verbose,
        )

        if self.box_size is not None:
            results = pbc_wrap_coords(results, self.box_size)
        
        if return_cpu:
            results = results.cpu()
        
        return results
