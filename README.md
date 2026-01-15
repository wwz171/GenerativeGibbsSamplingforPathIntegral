# GG-PI: Generative Gibbs Sampling for Path Integral

This repository contains the official implementation of **GG-PI** from the paper:

> **[Quantum Statistics from Classical Simulations via Generative Gibbs Sampling](<Add DOI or arXiv link>)**  
> *Weizhou Wang* and *Xuanxi Zhang*

GG-PI accelerates path integral molecular dynamics (PIMD) simulations using generative flow matching models with Gibbs sampling. This repository provides:
- Pre-trained model checkpoints
- Example datasets (MD and PIMD trajectories)
- Jupyter notebook tutorials for training and sampling
- Source code for neural networks, sampling algorithms, and simulation tools
---

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment with all dependencies
conda env create -f environment.yml
conda activate ggpi

# Install the ggpi package in editable mode
pip install -e .
```

### Option 2: Install dependencies only

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

### Verify Installation

```python
import ggpi
print(ggpi.__version__)

# Test imports
from ggpi.nn import ViSNet_Velocity_Network
from ggpi.sampling import ConditionalFlowSampler, GibbsSampler
from ggpi.simulation import SilveraGoldmanPotential
```

---

## Requirements

**Core Dependencies:**
- Python >= 3.10
- PyTorch >= 2.0.0 (with CUDA support recommended)
- PyTorch Geometric >= 2.3.0
- NumPy >= 1.21.0
- Numba >= 0.56.0
- tqdm >= 4.60.0
- Matplotlib >= 3.5.0

**For Analysis:**
- freud-analysis >= 2.0.0 (for RDF calculations)

**For Notebooks:**
- Jupyter
- ipykernel

---

## Repository Structure

```
GenerativeGibbsforPathIntegral/
├── src/ggpi/                  # Main package source code
│   ├── __init__.py            # Package initialization and exports
│   ├── nn/                    # Neural network architectures
│   │   └── visnet.py          # ViSNet-based velocity prediction network
│   ├── sampling/              # Sampling algorithms
│   │   ├── flow_matching.py   # Flow matching training and sampling
│   │   └── gibbs.py           # Gibbs sampler for path integral beads
│   ├── simulation/            # MD simulation tools
│   │   └── parah2.py          # Silvera-Goldman potential & Langevin integrator
│   └── utils/                 # Utility functions
│       ├── pbc.py             # Periodic boundary condition helpers
│       └── data.py            # Dataset classes and data processing
│
├── checkpoints/               # Pre-trained model weights
│   ├── parah2.pt              # Para-H2 (256 molecules, 100K)
│   ├── water.pt               # Water (216 molecules, 300K)
│   └── zundel.pt              # Zundel cation (H5O2+, 300K)
│
├── data/                      # Example datasets
│   ├── *_md_*.pkl             # Classical MD trajectories
│   └── *_pimd_*P32.pkl        # PIMD reference data (32 beads)
│
├── examples/                  # Jupyter notebook tutorials
│   ├── parah2_training.ipynb  # Training walkthrough
│   ├── parah2_sampling.ipynb  # Sampling and RDF analysis
│   ├── water.ipynb            # Water system demo
│   └── zundel.ipynb           # Zundel cation demo
│
├── others/                    # Reference files and external tool inputs
│   ├── parah2/                # i-PI/LAMMPS configs for para-H2
│   ├── water/                 # Water simulation configs
│   └── zundel/                # Zundel cation configs
│
├── pyproject.toml             # Package configuration (PEP 517/518)
├── requirements.txt           # pip dependencies
├── environment.yml            # Conda environment specification
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## Quick Start

### Load a Pre-trained Model and Sample

```python
import torch
from ggpi.nn import ViSNet_Velocity_Network
from ggpi.sampling import ConditionalFlowSampler, GibbsSampler

# Load pre-trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ViSNet_Velocity_Network.load_model('checkpoints/parah2.pt', device=device)

# Setup sampler
Z = torch.ones(256, dtype=torch.long)  # Atomic numbers (H2 -> 1)
prior_sigma = torch.tensor([0.5, 0.5, 0.5])  # Prior standard deviation
box_size = torch.tensor([21.06, 21.06, 21.06])  # Box dimensions in Bohr

sampler = ConditionalFlowSampler(
    model, Z, prior_sigma,
    box_size=box_size,
    device=device,
    num_steps=10
)

# Create Gibbs sampler
gibbs = GibbsSampler(sampler, box_size=box_size, device=device)

# Run sampling from initial configuration
init_config = torch.randn(1, 256, 3) * 0.1  # (batch, atoms, xyz)
final_config, history = gibbs.sample(
    init_config,
    num_beads=32,
    max_iterations=100,
    verbose=True
)
```

### Training a New Model

See [`examples/parah2_training.ipynb`](examples/parah2_training.ipynb) for a complete training tutorial.

---

## Examples

| Notebook | Description |
|----------|-------------|
| [`parah2_training.ipynb`](examples/parah2_training.ipynb) | Train a flow matching model for para-H2 |
| [`parah2_sampling.ipynb`](examples/parah2_sampling.ipynb) | Generate samples and compute RDF |
| [`water.ipynb`](examples/water.ipynb) | Water system sampling with O-O, O-H, H-H RDFs |
| [`zundel.ipynb`](examples/zundel.ipynb) | Zundel cation (H₅O₂⁺) proton transfer analysis |

---

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@article{wang2025ggpi,
  title   = {<Add paper title>},
  author  = {Wang, Weizhou and Zhang, Xuanxi},
  journal = {<Add venue>},
  year    = {2025},
  doi     = {<Add DOI>},
  url     = {<Add URL or arXiv link>}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
