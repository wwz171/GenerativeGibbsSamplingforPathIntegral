"""ViSNet-based velocity prediction network.

This module implements a tailored ViSNet architecture for velocity prediction
in molecular systems with periodic boundary conditions support.

Reference:
    Wang et al. "Enhancing Geometric Representations for Molecules with
    Equivariant Vector-Scalar Interactive Message Passing"
    https://doi.org/10.1038/s41467-023-43720-2
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Embedding, LayerNorm, Linear, Parameter

try:
    from torch_geometric.nn import MessagePassing, radius_graph
    from torch_geometric.typing import Adj, OptTensor
    from torch_geometric.utils import scatter
except ImportError:
    raise ImportError(
        "PyTorch Geometric is required for this module. "
        "Install it with: pip install torch-geometric"
    )

from ..utils.pbc import apply_minimum_image, compute_pbc_distances


# =============================================================================
# Graph Construction Utilities
# =============================================================================

def radius_graph_pbc(
    pos: Tensor,
    r_cut: float,
    box_size: Tensor = None,
    loop: bool = False
) -> Tensor:
    """Build radius graph with periodic boundary conditions.
    
    Args:
        pos: Node positions of shape (N, 3).
        r_cut: Cutoff radius.
        box_size: Box dimensions of shape (3,), or None for non-periodic.
        loop: Whether to include self-loops.
        
    Returns:
        Edge indices of shape (2, E).
    """
    N = pos.size(0)
    device = pos.device

    if N == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    if box_size is not None:
        box_size = box_size.to(device)

    pos_i = pos.unsqueeze(1)
    pos_j = pos.unsqueeze(0)

    if box_size is None:
        distances = torch.norm(pos_i - pos_j, dim=-1)
    else:
        distances = compute_pbc_distances(pos_i, pos_j, box_size)

    mask = distances <= r_cut
    if not loop:
        mask.fill_diagonal_(False)

    edge_indices = torch.nonzero(mask, as_tuple=False)

    if edge_indices.size(0) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    return edge_indices.t()


def build_radius_graph_pbc(
    pos: Tensor,
    r: float,
    batch: Optional[Tensor] = None,
    box_size: Optional[Tensor] = None,
) -> Tensor:
    """Build radius graph with PBC for batched data.
    
    Args:
        pos: Node positions of shape (N_total, 3).
        r: Cutoff radius.
        batch: Batch assignment of shape (N_total,).
        box_size: Box dimensions of shape (3,).
        
    Returns:
        Edge indices of shape (2, E).
    """
    device = pos.device
    N = pos.shape[0]
    
    if N == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    if batch is None:
        batch = torch.zeros(N, dtype=torch.long, device=device)
    
    if box_size is not None:
        if box_size.dim() == 0:
            box_size = box_size.expand(3)
        elif box_size.shape[0] != 3:
            raise ValueError("box_size must be scalar or 3-element tensor")
        box_size = box_size.to(device)
    
    edges = []
    unique_batches = torch.unique(batch)
    
    for b in unique_batches:
        mask = batch == b
        batch_pos = pos[mask]
        batch_indices = torch.where(mask)[0]
        n_atoms = batch_pos.shape[0]
        
        if n_atoms == 0:
            continue
        
        batch_edge_index = radius_graph_pbc(batch_pos, r, box_size, loop=False)
        
        if batch_edge_index.shape[1] == 0:
            continue
            
        if batch_edge_index.shape[1] > 0:
            global_src = batch_indices[batch_edge_index[0]]
            global_dst = batch_indices[batch_edge_index[1]]
            batch_edges = torch.stack([global_src, global_dst])
            edges.append(batch_edges)
    
    if edges:
        edge_index = torch.cat(edges, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    return edge_index


def compute_relative_coords(
    coors: Tensor,
    edge_index: Tensor,
    box_size: Tensor
) -> Tensor:
    """Compute relative coordinates with PBC.
    
    Args:
        coors: Node coordinates of shape (N, 3).
        edge_index: Edge indices of shape (2, E).
        box_size: Box dimensions of shape (3,).
        
    Returns:
        Relative coordinates of shape (E, 3).
    """
    rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
    if box_size is not None:
        rel_coors = apply_minimum_image(rel_coors, box_size)
    return rel_coors


# =============================================================================
# Helper Modules
# =============================================================================

class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal time embedding for diffusion/flow models."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Time values of shape (B,).
            
        Returns:
            Embeddings of shape (B, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        return emb


class Swish_(nn.Module):
    """Swish activation function (fallback for older PyTorch)."""
    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff function for distance-based interactions.
    
    Args:
        cutoff: Cutoff distance.
    """
    
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        """Apply cosine cutoff to distances.
        
        Args:
            distances: Distance values.
            
        Returns:
            Cutoff values in [0, 1].
        """
        cutoffs = 0.5 * ((distances * math.pi / self.cutoff).cos() + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    """Exponential normal radial basis function expansion.
    
    Args:
        cutoff: Cutoff distance.
        num_rbf: Number of radial basis functions.
        trainable: Whether RBF parameters are trainable.
    """
    
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 128,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter('means', Parameter(means))
            self.register_parameter('betas', Parameter(betas))
        else:
            self.register_buffer('means', means)
            self.register_buffer('betas', betas)

    def _initial_params(self) -> Tuple[Tensor, Tensor]:
        start_value = torch.exp(torch.tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value))**-2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        """Apply exponential normal smearing to distances.
        
        Args:
            dist: Distance values.
            
        Returns:
            RBF-expanded features.
        """
        dist = dist.unsqueeze(-1)
        smeared_dist = self.cutoff_fn(dist) * (-self.betas * (
            (self.alpha * (-dist)).exp() - self.means)**2).exp()
        return smeared_dist


class Sphere(nn.Module):
    """Spherical harmonics computation for edge vectors.
    
    Args:
        lmax: Maximum degree of spherical harmonics (1 or 2).
    """
    
    def __init__(self, lmax: int = 2) -> None:
        super().__init__()
        self.lmax = lmax

    def forward(self, edge_vec: Tensor) -> Tensor:
        """Compute spherical harmonics of edge vectors.
        
        Args:
            edge_vec: Edge vectors of shape (E, 3).
            
        Returns:
            Spherical harmonic features.
        """
        return self._spherical_harmonics(
            self.lmax,
            edge_vec[..., 0],
            edge_vec[..., 1],
            edge_vec[..., 2],
        )

    @staticmethod
    def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            ], dim=-1)

        raise ValueError(f"'lmax' needs to be 1 or 2 (got {lmax})")


class VecLayerNorm(nn.Module):
    """Layer normalization for vector features.
    
    Args:
        hidden_channels: Number of hidden channels.
        trainable: Whether normalization weights are trainable.
        norm_type: Type of normalization ('max_min' or None).
    """
    
    def __init__(
        self,
        hidden_channels: int,
        trainable: bool,
        norm_type: Optional[str] = 'max_min',
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.eps = 1e-12

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter('weight', Parameter(weight))
        else:
            self.register_buffer('weight', weight)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def max_min_norm(self, vec: Tensor) -> Tensor:
        dist = torch.norm(vec, dim=1, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = dist.max(dim=-1)
        min_val, _ = dist.min(dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1)) / delta.view(-1, 1)

        return dist.relu() * direct

    def forward(self, vec: Tensor) -> Tensor:
        if vec.size(1) == 3:
            if self.norm_type == 'max_min':
                vec = self.max_min_norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)

        raise ValueError(f"'{self.__class__.__name__}' only supports 3 or 8 "
                         f"channels (got {vec.size(1)})")


# =============================================================================
# ViSNet Message Passing Layers
# =============================================================================

class NeighborEmbedding(MessagePassing):
    """Neighborhood embedding module from ViSNet.
    
    Args:
        hidden_channels: Number of hidden channels.
        num_rbf: Number of radial basis functions.
        cutoff: Cutoff distance.
        max_z: Maximum atomic number.
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        max_z: int = 100,
    ) -> None:
        super().__init__(aggr='add')
        self.embedding = Embedding(max_z, hidden_channels)
        self.distance_proj = Linear(num_rbf, hidden_channels)
        self.combine = Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.zero_()
        self.combine.bias.data.zero_()

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class EdgeEmbedding(nn.Module):
    """Edge embedding module from ViSNet.
    
    Args:
        num_rbf: Number of radial basis functions.
        hidden_channels: Number of hidden channels.
    """
    
    def __init__(self, num_rbf: int, hidden_channels: int) -> None:
        super().__init__()
        self.edge_proj = Linear(num_rbf, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.zero_()

    def forward(
        self,
        edge_index: Tensor,
        edge_attr: Tensor,
        x: Tensor,
    ) -> Tensor:
        x_j = x[edge_index[0]]
        x_i = x[edge_index[1]]
        return (x_i + x_j) * self.edge_proj(edge_attr)


class ViS_MP(MessagePassing):
    """ViSNet message passing layer without vertex geometric features.
    
    Args:
        num_heads: Number of attention heads.
        hidden_channels: Number of hidden channels.
        cutoff: Cutoff distance.
        vecnorm_type: Type of vector normalization.
        trainable_vecnorm: Whether vector normalization is trainable.
        last_layer: Whether this is the last layer.
    """
    
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str],
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ) -> None:
        super().__init__(aggr='add', node_dim=0)

        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()
        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = Linear(hidden_channels, hidden_channels * 3, False)

        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels)
        self.dk_proj = Linear(hidden_channels, hidden_channels)
        self.dv_proj = Linear(hidden_channels, hidden_channels)

        self.s_proj = Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = Linear(hidden_channels, hidden_channels)
            self.w_src_proj = Linear(hidden_channels, hidden_channels, False)
            self.w_trg_proj = Linear(hidden_channels, hidden_channels, False)

        self.o_proj = Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec: Tensor, d_ij: Tensor) -> Tensor:
        """Compute component of vec orthogonal to d_ij."""
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.zero_()
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.zero_()
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.zero_()
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.zero_()
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.zero_()

        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.zero_()
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.zero_()
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij))
        dk = dk.reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij))
        dv = dv.reshape(-1, self.num_heads, self.head_dim)

        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )
        vec_dot = (vec1 * vec2).sum(dim=1)
    
        x, vec_out = self.propagate(
            edge_index, q=q, k=k, v=v, dk=dk, dv=dv,
            vec=vec, r_ij=r_ij, d_ij=d_ij
        )

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        
        if not self.last_layer:
            df_ij = self.edge_updater(
                edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij
            )
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_j: Tensor,
        dk: Tensor,
        dv: Tensor,
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(
            self.act(self.s_proj(v_j)), self.hidden_channels, dim=1
        )
        
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec_j

    def edge_update(
        self,
        vec_i: Tensor,
        vec_j: Tensor,
        d_ij: Tensor,
        f_ij: Tensor,
    ) -> Tensor:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class ViS_MP_Vertex(ViS_MP):
    """ViSNet message passing layer with vertex geometric features."""
    
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str],
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ) -> None:
        super().__init__(
            num_heads, hidden_channels, cutoff, vecnorm_type,
            trainable_vecnorm, last_layer
        )

        if not self.last_layer:
            self.f_proj = Linear(hidden_channels, hidden_channels * 2)
            self.t_src_proj = Linear(hidden_channels, hidden_channels, False)
            self.t_trg_proj = Linear(hidden_channels, hidden_channels, False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        if not self.last_layer:
            if hasattr(self, 't_src_proj'):
                nn.init.xavier_uniform_(self.t_src_proj.weight)
            if hasattr(self, 't_trg_proj'):
                nn.init.xavier_uniform_(self.t_trg_proj.weight)

    def edge_update(
        self,
        vec_i: Tensor,
        vec_j: Tensor,
        d_ij: Tensor,
        f_ij: Tensor,
    ) -> Tensor:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)

        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)

        f1, f2 = torch.split(
            self.act(self.f_proj(f_ij)), self.hidden_channels, dim=-1
        )

        return f1 * w_dot + f2 * t_dot


class GatedEquivariantBlock(nn.Module):
    """Gated equivariant block for output prediction.
    
    Args:
        hidden_channels: Number of input hidden channels.
        out_channels: Number of output channels.
        intermediate_channels: Number of intermediate channels.
        scalar_activation: Whether to apply activation to scalar output.
    """
    
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        scalar_activation: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.SiLU() if scalar_activation else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.zero_()

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)

        return x, v


class EquivariantScalar(nn.Module):
    """Compute final scalar outputs from node and vector features.
    
    Args:
        hidden_channels: Number of hidden channels.
    """
    
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()

        self.output_network = nn.ModuleList([
            GatedEquivariantBlock(
                hidden_channels,
                hidden_channels // 2,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(
                hidden_channels // 2,
                1,
                scalar_activation=False,
            ),
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x: Tensor, v: Tensor) -> Tensor:
        for layer in self.output_network:
            x, v = layer(x, v)
        return x + v.sum() * 0


# =============================================================================
# Main ViSNet Block
# =============================================================================

class ViSNetBlock_PBC(nn.Module):
    """ViSNet representation block with PBC support.
    
    Args:
        lmax: Maximum spherical harmonic degree (1 or 2).
        vecnorm_type: Type of vector normalization.
        trainable_vecnorm: Whether vector normalization is trainable.
        num_heads: Number of attention heads.
        num_layers: Number of message passing layers.
        hidden_channels: Number of hidden channels.
        num_rbf: Number of radial basis functions.
        trainable_rbf: Whether RBF parameters are trainable.
        max_z: Maximum atomic number.
        cutoff: Cutoff distance.
        vertex: Whether to use vertex geometric features.
    """
    
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 1,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 7.0,
        vertex: bool = False,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.trainable_rbf = trainable_rbf
        self.max_z = max_z
        self.cutoff = cutoff

        self.embedding = Embedding(max_z, hidden_channels)
        self.sphere = Sphere(lmax=lmax)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(
            hidden_channels, num_rbf, cutoff, max_z
        )
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels)

        self.vis_mp_layers = nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
        )
        vis_mp_class = ViS_MP if not vertex else ViS_MP_Vertex
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs)
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(vis_mp_class(last_layer=True, **vis_mp_kwargs))

        self.out_norm = LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(
        self,
        pos: Tensor,
        z: Tensor,
        h: Tensor,
        vec: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Adj = None,
        box_size: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x = h

        edge_vec = compute_relative_coords(pos, edge_index, box_size)
        edge_weight = torch.norm(edge_vec, dim=-1)

        if edge_attr is None:
            edge_attr = self.distance_expansion(edge_weight)

        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(
            edge_vec[mask], dim=1
        ).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        if vec is None:
            vec = torch.zeros(
                x.size(0), ((self.lmax + 1)**2) - 1, x.size(1),
                dtype=x.dtype, device=x.device
            )

        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(
                x, vec, edge_index, edge_weight, edge_attr, edge_vec
            )
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](
            x, vec, edge_index, edge_weight, edge_attr, edge_vec
        )
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)

        return x, vec


# =============================================================================
# Main Velocity Network
# =============================================================================

class ViSNet_Velocity_Network(nn.Module):
    """ViSNet-based velocity prediction network for flow matching.
    
    This network predicts velocities for conditional flow matching
    in path integral molecular dynamics.
    
    Args:
        n_layers: Number of ViSNet blocks.
        hidden_channels: Number of hidden channels.
        time_embedding_dim: Dimension of time embedding.
        equivariant_feats_dim: Dimension of equivariant features.
        pos_dim: Dimension of positions (typically 3).
        max_z: Maximum atomic number.
        vecnorm_type: Type of vector normalization.
        trainable_vecnorm: Whether vector normalization is trainable.
        num_heads: Number of attention heads.
        num_rbf: Number of radial basis functions.
        trainable_rbf: Whether RBF parameters are trainable.
        vertex: Whether to use vertex geometric features.
        cutoff: Cutoff distance for graph construction.
        encode_rela_dis: Whether to encode relative displacement magnitude.
    """

    def __init__(
        self,
        n_layers: int = 3,
        hidden_channels: int = 32,
        time_embedding_dim: int = 16,
        equivariant_feats_dim: int = 3,
        pos_dim: int = 3,
        max_z: int = 100,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        vertex: bool = False,
        cutoff: float = 7.0,
        encode_rela_dis: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.max_z = max_z
        self.time_embedding_dim = time_embedding_dim
        self.equivariant_feats_dim = equivariant_feats_dim
        self.pos_dim = pos_dim
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_rbf = num_rbf
        self.trainable_rbf = trainable_rbf
        self.vertex = vertex
        self.cutoff = cutoff
        self.encode_rela_dis = encode_rela_dis

        self.time_embed = SinusoidalTimeEmbed(time_embedding_dim)

        self.visnet_block = nn.ModuleList()
        for _ in range(n_layers):
            self.visnet_block.append(
                ViSNetBlock_PBC(
                    lmax=1,
                    vecnorm_type=vecnorm_type,
                    trainable_vecnorm=trainable_vecnorm,
                    num_heads=num_heads,
                    hidden_channels=hidden_channels,
                    num_rbf=num_rbf,
                    trainable_rbf=trainable_rbf,
                    max_z=max_z,
                    cutoff=cutoff,
                    vertex=vertex,
                    num_layers=1,
                )
            )

        self.z_embd = Embedding(max_z, hidden_channels)
        self.vec_input_proj = nn.Linear(1, hidden_channels, bias=False)
        self.vec_output_proj = nn.Linear(hidden_channels, 1, bias=False)

        if encode_rela_dis:
            h_input_dim = hidden_channels + time_embedding_dim + 1
        else:
            h_input_dim = hidden_channels + time_embedding_dim

        self.h_input_proj = nn.Sequential(
            nn.Linear(h_input_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def _build_graph(
        self,
        pos: Tensor,
        batch: Optional[Tensor],
        box_size: Optional[Tensor] = None
    ) -> Tensor:
        """Build radius graph for the input positions."""
        try:
            if box_size is None:
                return radius_graph(pos, r=self.cutoff, batch=batch)
            else:
                return build_radius_graph_pbc(pos, self.cutoff, batch, box_size)
        except Exception as e:
            print(f"Warning: Graph construction failed: {e}")
            print("Falling back to manual implementation")
            return build_radius_graph_pbc(pos, self.cutoff, batch, box_size)

    def forward(
        self,
        pos: Tensor,
        z: Tensor,
        vec: Tensor,
        t: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        box_size: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass for velocity prediction.
        
        Args:
            pos: Node positions of shape (B, N, 3) or (N, 3).
            z: Atomic numbers of shape (B, N) or (N,).
            vec: Input vector features (displacement) of shape (B, N, 3) or (N, 3).
            t: Time values of shape (B,) or (B*N,).
            edge_index: Optional pre-computed edge indices.
            edge_attr: Optional edge attributes.
            batch: Batch assignment vector.
            box_size: Box dimensions for PBC.
            
        Returns:
            h_out: Output scalar features.
            vec_out: Output velocity (equivariant).
            edge_index: Edge indices used.
        """
        is_batched = (len(pos.shape) == 3)
        
        if is_batched:
            B, N, D = pos.shape
            original_v_shape = vec.shape

            pos_flatten = pos.reshape(-1, D)
            z_flatten = z.reshape(-1)
            vec_flatten = vec.reshape(-1, original_v_shape[-1])

            if batch is None:
                batch = torch.arange(B, device=pos.device).repeat_interleave(N)
            
            if t.dim() == 1 and len(t) == B:
                t_nodes = t.repeat_interleave(N)
            elif t.dim() == 1 and len(t) == B * N:
                t_nodes = t
            else:
                raise ValueError(
                    f"Invalid time tensor shape: {t.shape}. "
                    "Expected shape (B,) or (B*N,)."
                )
        
            pos_work = pos_flatten
            z_work = z_flatten
            vec_work = vec_flatten
            t_work = t_nodes
        else:
            pos_work = pos
            z_work = z
            vec_work = vec
            t_work = t

            if t_work.dim() == 1 and len(t_work) != pos_work.shape[0]:
                if batch is not None:
                    t_work = t_work[batch]
                else:
                    t_work = t_work[0].repeat(pos_work.shape[0])

        t_emb = self.time_embed(t_work)
        z_emb = self.z_embd(z_work)
        
        if self.encode_rela_dis:
            rela_vec = torch.norm(vec_work, dim=-1, keepdim=True)
            h_input = torch.cat([z_emb, t_emb, rela_vec], dim=-1)
        else:
            h_input = torch.cat([z_emb, t_emb], dim=-1)

        h_work = self.h_input_proj(h_input)
        
        if vec_work.ndim == 2:
            vec_work = vec_work.unsqueeze(-1)
            
        if edge_index is None:
            edge_index = self._build_graph(pos_work, batch, box_size)
            edge_index = edge_index.to(pos_work.device)

        vec_work = self.vec_input_proj(vec_work)

        for visnet_block in self.visnet_block:
            h_work, vec_work = visnet_block(
                pos_work, z_work, h_work, vec_work, edge_index,
                edge_attr=edge_attr, batch=batch, box_size=box_size
            )

        vec_work = self.vec_output_proj(vec_work)

        if is_batched:
            h_out = h_work.reshape(B, N, -1)
            vec_out = vec_work.reshape(B, N, 3, -1)
            vec_out = vec_out.mean(dim=-1)
        else:
            h_out = h_work
            vec_out = vec_work.squeeze(-1)
        
        return h_out, vec_out, edge_index

    def get_config(self) -> dict:
        """Return model configuration for serialization."""
        return {
            'n_layers': self.n_layers,
            'hidden_channels': self.hidden_channels,
            'time_embedding_dim': self.time_embedding_dim,
            'equivariant_feats_dim': self.equivariant_feats_dim,
            'pos_dim': self.pos_dim,
            'max_z': self.max_z,
            'vecnorm_type': self.vecnorm_type,
            'trainable_vecnorm': self.trainable_vecnorm,
            'num_heads': self.num_heads,
            'num_rbf': self.num_rbf,
            'trainable_rbf': self.trainable_rbf,
            'vertex': self.vertex,
            'cutoff': self.cutoff,
            'encode_rela_dis': self.encode_rela_dis,
        }
    
    def save_model(self, path: str):
        """Save model weights and configuration.
        
        Args:
            path: Path to save the checkpoint.
        """
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config(),
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu') -> 'ViSNet_Velocity_Network':
        """Load model from checkpoint.
        
        Args:
            path: Path to the checkpoint file.
            device: Device to load the model to.
            
        Returns:
            Loaded model instance.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device)
