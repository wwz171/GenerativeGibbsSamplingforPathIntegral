"""GGPI: Generative Gibbs Sampling for Path Integral Molecular Dynamics.

This package provides tools for accelerating path integral molecular dynamics
(PIMD) simulations using generative models with Gibbs sampling.

Main components:
    - nn: Neural network architectures (ViSNet-based velocity prediction)
    - sampling: Samplers (Flow matching, Gibbs sampling)
    - simulation: MD simulators (Para-H2 with Silvera-Goldman potential)
    - utils: Utility functions (PBC handling, data processing)

Example usage:
    >>> from ggpi.nn import ViSNet_Velocity_Network
    >>> from ggpi.sampling import ConditionalFlowSampler, GibbsSampler
    >>> 
    >>> # Load pre-trained model
    >>> model = ViSNet_Velocity_Network.load_model('checkpoints/parah2.pt')
    >>> 
    >>> # Create sampler
    >>> sampler = ConditionalFlowSampler(model, Z, prior_sigma, box_size=box)
    >>> 
    >>> # Run Gibbs sampling
    >>> gibbs = GibbsSampler(sampler, box_size=box, device='cuda')
    >>> final_config, history = gibbs.sample(init_data, num_beads=32, max_iterations=100)
"""

__version__ = "0.1.0"

from .nn import ViSNet_Velocity_Network, ViSNetBlock_PBC
from .sampling import (
    ConditionalLinearFlowMatching,
    ConditionalFlowSampler,
    GibbsSampler,
)
from .simulation import (
    SilveraGoldmanPotential,
    LangevinIntegrator,
    MDConfig,
    calculate_omega,
)
from .utils import (
    MolecularDataset,
    get_training_tuples,
    apply_minimum_image,
    pbc_wrap_coords,
)

__all__ = [
    # Neural networks
    "ViSNet_Velocity_Network",
    "ViSNetBlock_PBC",
    # Sampling
    "ConditionalLinearFlowMatching",
    "ConditionalFlowSampler",
    "GibbsSampler",
    # Simulation
    "SilveraGoldmanPotential",
    "LangevinIntegrator",
    "MDConfig",
    "calculate_omega",
    # Utils
    "MolecularDataset",
    "get_training_tuples",
    "apply_minimum_image",
    "pbc_wrap_coords",
]
