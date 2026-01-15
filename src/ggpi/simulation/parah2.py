"""Para-H2 molecular dynamics simulator.

This module implements the Silvera-Goldman potential for para-hydrogen
and Langevin integrators for molecular dynamics simulations.

Reference:
    Silvera, I.F. and Goldman, V.V. (1978). The isotropic intermolecular
    potential for H2 and D2 in the solid and gas phases.
    J. Chem. Phys. 69, 4209.
"""

import numpy as np
import numba as nb
from multiprocessing import Pool
import os
from typing import Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


# =============================================================================
# Global Process Pool Management
# =============================================================================

_global_pool = None


def _init_global_pool(num_cpus: Optional[int] = None):
    """Initialize global process pool for parallel computation."""
    global _global_pool
    if _global_pool is None:
        if num_cpus is not None:
            n_cores = num_cpus
        else:
            n_cores = os.cpu_count()
        _global_pool = Pool(n_cores)
    return _global_pool


def _cleanup_global_pool():
    """Clean up global process pool."""
    global _global_pool
    if _global_pool is not None:
        _global_pool.close()
        _global_pool.join()
        _global_pool = None


# =============================================================================
# Silvera-Goldman Potential Parameters (Atomic Units)
# =============================================================================

ALPHA = 1.713
BETA = 1.5671
DELTA = 0.00993
RC_EXP = 8.32
C_6 = 12.14
C_8 = 215.2
C_9 = 143.1
C_10 = 4813.9

# Pre-computed constants for efficiency
DELTA_2 = DELTA * 2.0
C_6_DIFF = C_6 * 6.0
C_8_DIFF = C_8 * 8.0
C_9_DIFF = C_9 * 9.0
C_10_DIFF = C_10 * 10.0


# =============================================================================
# JIT-Compiled Potential Functions
# =============================================================================

@nb.njit(fastmath=False, cache=True)
def _damping_function_numba(r: float) -> Tuple[float, float]:
    """Compute damping function and its derivative."""
    if r > RC_EXP:
        return 1.0, 0.0
    else:
        dist_frac = RC_EXP / r - 1.0
        fc = np.exp(-dist_frac * dist_frac)
        fc_diff = 2.0 * dist_frac * RC_EXP * fc / (r * r)
        return fc, fc_diff


@nb.njit(fastmath=False, cache=True)
def _repulsive_part_numba(r: float) -> Tuple[float, float]:
    """Compute repulsive part of potential and force."""
    exp_arg = ALPHA - r * (BETA + DELTA * r)
    pot = np.exp(exp_arg)
    force_mag = (BETA + DELTA_2 * r) * pot
    return pot, force_mag


@nb.njit(fastmath=False, cache=True)
def _sg_pair_potential_numba(r: float) -> Tuple[float, float]:
    """Compute Silvera-Goldman pair potential and force magnitude."""
    if r < 1e-10:
        r = 1e-10
    
    onr = 1.0 / r
    onr3 = onr * onr * onr
    onr6 = onr3 * onr3
    onr8 = onr6 * onr * onr
    onr9 = onr8 * onr
    onr10 = onr9 * onr
    
    exp_pot, exp_force = _repulsive_part_numba(r)
    fc, fc_diff = _damping_function_numba(r)
    
    disp = -(C_6 * onr6 + C_8 * onr8 - C_9 * onr9 + C_10 * onr10)
    disp_diff = (
        C_6_DIFF * onr6 * onr + C_8_DIFF * onr8 * onr 
        - C_9_DIFF * onr9 * onr + C_10_DIFF * onr10 * onr
    )
    
    pot = exp_pot + disp * fc
    force_mag = exp_force - disp_diff * fc - disp * fc_diff
    
    return pot, force_mag


@nb.njit(fastmath=False, cache=True)
def _minimum_image_distance_numba_cubic(
    dx: float, dy: float, dz: float,
    cell_inv: np.ndarray, cell: np.ndarray
) -> Tuple[float, float, float]:
    """Apply minimum image convention for cubic box."""
    new_dx = dx - np.round(dx / cell[0, 0]) * cell[0, 0]
    new_dy = dy - np.round(dy / cell[1, 1]) * cell[1, 1]
    new_dz = dz - np.round(dz / cell[2, 2]) * cell[2, 2]
    return new_dx, new_dy, new_dz


@nb.njit(fastmath=False, cache=True)
def _compute_single_trajectory_numba(
    positions: np.ndarray,
    cutoff: float,
    cutoff_sq: float,
    use_pbc: bool,
    cell: np.ndarray,
    cell_inv: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Compute energy and forces for a single configuration."""
    natoms = positions.shape[0]
    energy = 0.0
    forces = np.zeros_like(positions)
    
    for i in range(natoms):
        for j in range(i + 1, natoms):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]

            if use_pbc:
                dx, dy, dz = _minimum_image_distance_numba_cubic(
                    dx, dy, dz, cell_inv, cell
                )
            
            r_sq = dx * dx + dy * dy + dz * dz
            
            if r_sq >= cutoff_sq:
                continue
            
            r = np.sqrt(r_sq)
            pot_ij, force_mag = _sg_pair_potential_numba(r)
            
            energy += pot_ij
            
            force_factor = force_mag / r
            fx = force_factor * dx
            fy = force_factor * dy
            fz = force_factor * dz
            
            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[i, 2] += fz
            forces[j, 0] -= fx
            forces[j, 1] -= fy
            forces[j, 2] -= fz
    
    return energy, forces


def _compute_trajectory_chunk(args):
    """Compute energies and forces for a chunk of trajectories."""
    positions_chunk, cutoff, cutoff_sq, use_pbc, cell, cell_inv = args
    ntrajs = positions_chunk.shape[0]
    
    energies = np.zeros(ntrajs)
    forces = np.zeros_like(positions_chunk)
    
    for traj_idx in range(ntrajs):
        energy, force = _compute_single_trajectory_numba(
            positions_chunk[traj_idx], cutoff, cutoff_sq, use_pbc, cell, cell_inv
        )
        energies[traj_idx] = energy
        forces[traj_idx] = force
    
    return energies, forces


# =============================================================================
# Silvera-Goldman Potential Class
# =============================================================================

class SilveraGoldmanPotential:
    """Silvera-Goldman potential for para-hydrogen.
    
    Args:
        cutoff: Cutoff distance in Bohr.
        cell: 3x3 cell matrix for PBC, or None for non-periodic.
        use_multiprocessing: Whether to use parallel computation.
        chunk_size: Size of trajectory chunks for parallel processing.
        num_cpus: Number of CPU cores to use.
    """
    
    def __init__(
        self,
        cutoff: float = 8.0,
        cell: Optional[np.ndarray] = None,
        use_multiprocessing: bool = True,
        chunk_size: Optional[int] = None,
        num_cpus: Optional[int] = 8,
    ):
        self.cutoff = float(cutoff)
        self.cutoff_sq = self.cutoff * self.cutoff
        
        self.use_pbc = cell is not None
        if self.use_pbc:
            self.cell = np.asarray(cell, dtype=np.float64)
            self.cell_inv = np.linalg.inv(self.cell)
        else:
            self.cell = np.eye(3, dtype=np.float64)
            self.cell_inv = np.eye(3, dtype=np.float64)
        
        self.use_multiprocessing = use_multiprocessing
        self.chunk_size = chunk_size if chunk_size is not None else max(1, os.cpu_count() * 2)
        
        if self.use_multiprocessing:
            _init_global_pool(num_cpus=num_cpus)
    
    def compute_forces_and_energy(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces and energies for given positions.
        
        Args:
            positions: Atomic positions of shape (N, 3) or (T, N, 3).
            
        Returns:
            Tuple of (energies, forces).
        """
        positions = np.asarray(positions, dtype=np.float64)
        
        if positions.ndim == 2:
            positions = positions[np.newaxis, ...]
            single_traj = True
        elif positions.ndim == 3:
            single_traj = False
        else:
            raise ValueError(f"Wrong input shape: {positions.ndim}D")
        
        ntrajs, natoms = positions.shape[:2]
        
        if ntrajs == 1 or not self.use_multiprocessing or ntrajs < self.chunk_size:
            energies = np.zeros(ntrajs)
            forces = np.zeros_like(positions)
            
            for traj_idx in range(ntrajs):
                energy, force = _compute_single_trajectory_numba(
                    positions[traj_idx], self.cutoff, self.cutoff_sq,
                    self.use_pbc, self.cell, self.cell_inv
                )
                energies[traj_idx] = energy
                forces[traj_idx] = force
        else:
            pool = _init_global_pool()
            
            chunks = []
            for start_idx in range(0, ntrajs, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, ntrajs)
                chunk_args = (
                    positions[start_idx:end_idx],
                    self.cutoff, self.cutoff_sq, self.use_pbc,
                    self.cell, self.cell_inv
                )
                chunks.append(chunk_args)
            
            results = pool.map(_compute_trajectory_chunk, chunks)
            
            energies = np.concatenate([result[0] for result in results])
            forces = np.concatenate([result[1] for result in results], axis=0)
        
        if single_traj:
            return energies[0], forces[0]
        else:
            return energies, forces
    
    def __call__(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces and energies (callable interface)."""
        return self.compute_forces_and_energy(positions)
    
    def set_cell(self, cell: np.ndarray):
        """Update the simulation cell."""
        self.cell = np.asarray(cell, dtype=np.float64)
        self.cell_inv = np.linalg.inv(self.cell)
        self.use_pbc = True


# =============================================================================
# Physical Constants
# =============================================================================

KB_ATOMIC = 3.1668e-6  # Boltzmann constant in Hartree/K
AU_TIME_PER_FS = 41.341  # 1 fs = 41.341 a.u. of time


@dataclass
class MDConfig:
    """Configuration for molecular dynamics simulations.
    
    Args:
        temperature: Temperature in Kelvin.
        friction: Friction coefficient in fs^-1.
        dt: Time step in fs.
        mass: Particle mass in electron masses (default: H2 mass).
    """
    temperature: float
    friction: float
    dt: float
    mass: float = 3672  # H2 mass in atomic units


# =============================================================================
# JIT-Compiled MD Helper Functions
# =============================================================================

@nb.njit(fastmath=False, cache=True)
def _maxwell_boltzmann_velocities_numba(
    natoms: int,
    temperature: float,
    mass: float,
    seed: int
) -> np.ndarray:
    """Generate Maxwell-Boltzmann distributed velocities."""
    np.random.seed(seed)
    sigma = np.sqrt(KB_ATOMIC * temperature / mass)
    velocities = np.random.normal(0.0, sigma, (natoms, 3))
    return velocities


@nb.njit(fastmath=False, cache=True)
def _apply_pbc_numba(
    positions: np.ndarray,
    box_size: np.ndarray
) -> np.ndarray:
    """Apply periodic boundary conditions."""
    return positions - box_size * np.floor(positions / box_size)


@nb.njit(fastmath=False, cache=True)
def _baoab_step_batch_numba(
    positions_batch: np.ndarray,
    velocities_batch: np.ndarray,
    forces_batch: np.ndarray,
    dt_au: float,
    gamma_au: float,
    c1: float,
    c2: float,
    mass: float,
    noise_sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform BAOAB Langevin integration step for batch."""
    ntrajs, natoms, _ = positions_batch.shape
    
    for traj in range(ntrajs):
        # B: Half kick
        velocities_batch[traj] += 0.5 * dt_au * forces_batch[traj] / mass
        
        # A: Position update
        positions_batch[traj] += 0.5 * dt_au * velocities_batch[traj]
        
        # O: Stochastic update
        for i in range(natoms):
            for j in range(3):
                random_force = np.random.normal(0.0, 1.0)
                velocities_batch[traj, i, j] = (
                    c1 * velocities_batch[traj, i, j]
                    + c2 * random_force * noise_sigma
                )
                
        # A: Second position update
        positions_batch[traj] += 0.5 * dt_au * velocities_batch[traj]
    
    return positions_batch, velocities_batch


@nb.njit(fastmath=False, cache=True)
def _harmonic_forces_numba(
    positions: np.ndarray,
    reference_positions: np.ndarray,
    omega_au: float,
    mass: float,
    box_size: np.ndarray = None
) -> np.ndarray:
    """Calculate harmonic restraint forces: F = -m*omega^2*(x-x0)."""
    displacement = positions - reference_positions
    if box_size is not None:
        displacement = displacement - box_size * np.round(displacement / box_size)

    force_constant = mass * omega_au * omega_au
    forces = -force_constant * displacement
    return forces


@nb.njit(fastmath=False, cache=True)
def _constrained_step_batch_numba(
    positions_batch: np.ndarray,
    velocities_batch: np.ndarray,
    forces_batch: np.ndarray,
    reference_batch: np.ndarray,
    dt_au: float,
    gamma_au: float,
    c1: float,
    c2: float,
    mass: float,
    noise_sigma: float,
    omega_au: float,
    box_size: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform constrained BAOAB step with harmonic restraints."""
    ntrajs, natoms, _ = positions_batch.shape
    
    for traj in range(ntrajs):
        harmonic_forces = _harmonic_forces_numba(
            positions_batch[traj], reference_batch[traj], omega_au, mass, box_size
        )
        total_forces = forces_batch[traj] + harmonic_forces
        
        # B: Half kick
        velocities_batch[traj] += 0.5 * dt_au * total_forces / mass
        
        # A: Position update
        positions_batch[traj] += 0.5 * dt_au * velocities_batch[traj]
        
        # O: Stochastic update
        for i in range(natoms):
            for j in range(3):
                random_force = np.random.normal(0.0, 1.0)
                velocities_batch[traj, i, j] = (
                    c1 * velocities_batch[traj, i, j]
                    + c2 * random_force * noise_sigma
                )
        
        # A: Second position update
        positions_batch[traj] += 0.5 * dt_au * velocities_batch[traj]
    
    return positions_batch, velocities_batch


@nb.njit(fastmath=False, cache=True)
def _calculate_kinetic_energy_numba(
    velocities: np.ndarray,
    mass: float
) -> np.ndarray:
    """Calculate kinetic energy for batch of trajectories."""
    ntrajs, natoms, _ = velocities.shape
    ke = np.zeros(ntrajs)
    
    for traj in range(ntrajs):
        for i in range(natoms):
            for j in range(3):
                ke[traj] += 0.5 * mass * velocities[traj, i, j]**2
    
    return ke


# =============================================================================
# Langevin Integrator Class
# =============================================================================

class LangevinIntegrator:
    """Langevin dynamics integrator using BAOAB scheme.
    
    Args:
        potential_func: Potential energy function.
        config: MD configuration parameters.
    """
    
    def __init__(self, potential_func, config: MDConfig):
        self.potential_func = potential_func
        self.config = config
        self.cutoff = potential_func.cutoff
        
        # Convert to atomic units
        self.dt_au = config.dt * AU_TIME_PER_FS
        self.gamma_au = config.friction / AU_TIME_PER_FS
        self.temperature_au = config.temperature * KB_ATOMIC
        
        # BAOAB coefficients
        self.c1 = np.exp(-self.gamma_au * self.dt_au)
        self.c2 = np.sqrt(1.0 - self.c1**2)
        self.noise_sigma = np.sqrt(self.temperature_au / config.mass)
        
        # State variables
        self.positions = None
        self.velocities = None
        self.forces = None
        self.box_size = None
        self.current_energy = None
        
        print(f"Integrator initialized:")
        print(f"  dt: {config.dt} fs -> {self.dt_au:.6f} a.u.")
        print(f"  gamma: {config.friction} fs^-1 -> {self.gamma_au:.6f} a.u.")
        print(f"  T: {config.temperature} K -> {self.temperature_au:.6e} a.u.")
        print(f"  mass: {config.mass} m_e")
        
    def initialize_cubic_lattice(
        self,
        box_size: np.ndarray,
        spacing: float = 3.0,
        ntrajs: int = 1,
        random_displacement: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize system on a cubic lattice.
        
        Args:
            box_size: Box dimensions.
            spacing: Lattice spacing.
            ntrajs: Number of trajectories.
            random_displacement: Random displacement magnitude.
            
        Returns:
            Tuple of (positions, velocities).
        """
        self.box_size = np.array(box_size, dtype=np.float64)

        if self.box_size[0] < 2 * self.cutoff:
            raise ValueError(
                f"Box size {self.box_size} is too small for cutoff {self.cutoff}."
            )
        
        # Calculate lattice positions
        nx = int(box_size[0] / spacing)
        ny = int(box_size[1] / spacing)
        nz = int(box_size[2] / spacing)
        
        x = np.linspace(0, box_size[0] - spacing, nx)
        y = np.linspace(0, box_size[1] - spacing, ny)
        z = np.linspace(0, box_size[2] - spacing, nz)
        
        positions_list = []
        for xi in x:
            for yi in y:
                for zi in z:
                    positions_list.append([xi, yi, zi])
        
        natoms = len(positions_list)
        
        # Create trajectory copies
        self.positions = np.zeros((ntrajs, natoms, 3))
        for traj in range(ntrajs):
            base_positions = np.array(positions_list)
            
            if random_displacement > 0:
                displacement = np.random.normal(0, random_displacement, (natoms, 3))
                base_positions += displacement
            
            self.positions[traj] = _apply_pbc_numba(base_positions, self.box_size)
        
        # Initialize velocities
        self.velocities = np.zeros((ntrajs, natoms, 3))
        for traj in range(ntrajs):
            seed = np.random.randint(0, 1000000) + traj
            self.velocities[traj] = _maxwell_boltzmann_velocities_numba(
                natoms, self.config.temperature, self.config.mass, seed
            )
        
        # Remove COM motion
        for traj in range(ntrajs):
            com_velocity = np.mean(self.velocities[traj], axis=0)
            self.velocities[traj] -= com_velocity
        
        self._update_forces()
        return self.positions.copy(), self.velocities.copy()
    
    def _update_forces(self):
        """Update forces from potential."""
        if self.box_size is not None:
            cell = np.diag(self.box_size)
            self.potential_func.set_cell(cell)
        
        energies, forces = self.potential_func(self.positions)
        self.forces = forces
        self.current_energy = energies
    
    def run(
        self,
        nsteps: int,
        sample_interval: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run MD simulation.
        
        Args:
            nsteps: Number of simulation steps.
            sample_interval: Interval for saving frames.
            
        Returns:
            Tuple of (trajectory, energies, temperatures, kinetic_energies).
        """
        if self.positions is None:
            raise ValueError("Must initialize system first")
        
        ntrajs, natoms, _ = self.positions.shape
        nsamples = nsteps // sample_interval
        
        # Pre-allocate arrays
        trajectory = np.zeros((ntrajs, nsamples, natoms, 3))
        energies = np.zeros((ntrajs, nsamples))
        kinetic_energy = np.zeros((ntrajs, nsamples))
        ins_temperature = np.zeros((ntrajs, nsamples))
        
        sample_idx = 0
        
        for step in tqdm(range(nsteps), desc="MD simulation"):
            # BAOAB step
            self.positions, self.velocities = _baoab_step_batch_numba(
                self.positions, self.velocities, self.forces,
                self.dt_au, self.gamma_au, self.c1, self.c2,
                self.config.mass, self.noise_sigma
            )

            # Apply PBC
            if self.box_size is not None:
                for traj in range(ntrajs):
                    self.positions[traj] = _apply_pbc_numba(
                        self.positions[traj], self.box_size
                    )
            
            # Update forces
            self._update_forces()
            
            # Second half kick
            self.velocities += 0.5 * self.dt_au * self.forces / self.config.mass
            
            # Sample
            if (step + 1) % sample_interval == 0:
                trajectory[:, sample_idx] = self.positions.copy()
                kinetic_energy[:, sample_idx] = self.get_kinetic_energy().copy()
                energies[:, sample_idx] = self.current_energy.copy()
                ins_temperature[:, sample_idx] = self.get_temperature().copy()
                sample_idx += 1
        
        return trajectory, energies, ins_temperature, kinetic_energy

    def run_constrained_sampling(
        self,
        reference_positions: np.ndarray,
        omega_fs: float,
        nsteps: int,
        temperature: Optional[float] = None,
        friction: Optional[float] = None,
        dt: Optional[float] = None,
        sample_interval: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run constrained sampling with harmonic restraints.
        
        Args:
            reference_positions: Reference positions for restraints.
            omega_fs: Harmonic frequency in fs^-1.
            nsteps: Number of simulation steps.
            temperature: Temperature (uses config default if None).
            friction: Friction (uses config default if None).
            dt: Time step (uses config default if None).
            sample_interval: Interval for saving frames.
            
        Returns:
            Tuple of (trajectory, energies, temperatures).
        """
        if reference_positions.ndim == 4:
            reference_positions = reference_positions.reshape(
                -1, reference_positions.shape[-2], 3
            )
        if reference_positions.ndim != 3:
            raise ValueError(
                f"Expected 3D array (nsamples, natoms, 3), "
                f"got {reference_positions.ndim}D"
            )
        
        nsamples, natoms, _ = reference_positions.shape
        
        # Update parameters if provided
        original_config = self.config
        if temperature is not None or friction is not None or dt is not None:
            new_config = MDConfig(
                temperature=temperature if temperature is not None else original_config.temperature,
                friction=friction if friction is not None else original_config.friction,
                dt=dt if dt is not None else original_config.dt,
                mass=original_config.mass
            )
            
            dt_au = new_config.dt * AU_TIME_PER_FS
            gamma_au = new_config.friction / AU_TIME_PER_FS
            temperature_au = new_config.temperature * KB_ATOMIC
            
            c1 = np.exp(-gamma_au * dt_au)
            c2 = np.sqrt(1.0 - c1**2)
            noise_sigma = np.sqrt(temperature_au / new_config.mass)
        else:
            new_config = original_config
            dt_au = self.dt_au
            gamma_au = self.gamma_au
            c1 = self.c1
            c2 = self.c2
            noise_sigma = self.noise_sigma
        
        omega_au = omega_fs / AU_TIME_PER_FS
        
        print(f"Constrained sampling:")
        print(f"  Starting points: {nsamples}")
        print(f"  Steps per point: {nsteps}")
        print(f"  Frequency: {omega_fs} fs^-1 -> {omega_au:.6f} a.u.")
        print(f"  Temperature: {new_config.temperature} K")
        
        n_sampled_frames = nsteps // sample_interval
        nframes = n_sampled_frames + 2
        
        # Pre-allocate output arrays
        trajectory = np.zeros((nsamples, nframes, natoms, 3))
        energies = np.zeros((nsamples, nframes))
        temperatures = np.zeros((nsamples, nframes))
        
        trajectory[:, 0] = reference_positions.copy()
        
        energies_pot_init, _ = self.potential_func(reference_positions)
        energies[:, 0] = energies_pot_init.copy()
        temperatures[:, 0] = new_config.temperature
        
        # Initialize velocities
        velocities_batch = np.zeros((nsamples, natoms, 3))
        for i in range(nsamples):
            seed = np.random.randint(0, 1000000) + i
            velocities_batch[i] = _maxwell_boltzmann_velocities_numba(
                natoms, new_config.temperature, new_config.mass, seed
            )
        
        positions_batch = reference_positions.copy()
        
        # Main simulation loop
        sample_idx = 1
        energies_pot, forces_pot = self.potential_func(positions_batch)
        
        for step in tqdm(range(nsteps), desc="Constrained sampling"):
            positions_batch, velocities_batch = _constrained_step_batch_numba(
                positions_batch, velocities_batch, forces_pot, reference_positions,
                dt_au, gamma_au, c1, c2, new_config.mass, noise_sigma, omega_au,
                self.box_size
            )
            
            if self.box_size is not None:
                for i in range(nsamples):
                    positions_batch[i] = _apply_pbc_numba(
                        positions_batch[i], self.box_size
                    )
            
            energies_pot, forces_pot = self.potential_func(positions_batch)
            
            # Second half kick with harmonic forces
            for i in range(nsamples):
                harmonic_forces = _harmonic_forces_numba(
                    positions_batch[i], reference_positions[i],
                    omega_au, new_config.mass, box_size=self.box_size
                )
                total_forces = forces_pot[i] + harmonic_forces
                velocities_batch[i] += 0.5 * dt_au * total_forces / new_config.mass
            
            if (step + 1) % sample_interval == 0 and sample_idx < nframes - 1:
                trajectory[:, sample_idx] = positions_batch.copy()
                energies[:, sample_idx] = energies_pot.copy()
                
                ke = _calculate_kinetic_energy_numba(velocities_batch, new_config.mass)
                temperatures[:, sample_idx] = (
                    2.0 * ke / (3.0 * natoms * KB_ATOMIC)
                ).clip(min=0.0)
                
                sample_idx += 1
        
        # Final frame
        trajectory[:, -1] = positions_batch.copy()
        energies[:, -1] = energies_pot.copy()
        ke = _calculate_kinetic_energy_numba(velocities_batch, new_config.mass)
        temperatures[:, -1] = (2.0 * ke / (3.0 * natoms * KB_ATOMIC)).clip(min=0.0)
        
        return trajectory, energies, temperatures

    def get_kinetic_energy(self) -> np.ndarray:
        """Get kinetic energy for all trajectories."""
        if self.velocities is None:
            return None
        return _calculate_kinetic_energy_numba(self.velocities, self.config.mass)
    
    def get_temperature(self) -> np.ndarray:
        """Get instantaneous temperature for all trajectories."""
        ke = self.get_kinetic_energy()
        if ke is None:
            return None
        
        natoms = self.positions.shape[1]
        return 2.0 * ke / (3.0 * natoms * KB_ATOMIC)


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_omega(T_kelvin: float, return_units: str = 'fs') -> float:
    """Calculate path integral bead frequency from temperature.
    
    Args:
        T_kelvin: Temperature in Kelvin.
        return_units: Output units ('fs', 'ps', or 's').
        
    Returns:
        Angular frequency omega.
    """
    if return_units not in ['fs', 'ps', 's']:
        raise ValueError("return_units must be 'fs', 'ps', or 's'")
    
    k_B_SI = 1.380649e-23  # Boltzmann constant in J/K
    hbar_SI = 1.054571817e-34  # Reduced Planck constant in J*s
    
    # omega = sqrt(2) * k_B * T / hbar
    w_SI = np.sqrt(2) * k_B_SI * T_kelvin / hbar_SI
    
    convert_factor = {
        'fs': 1e15,
        'ps': 1e12,
        's': 1.0
    }
    
    return w_SI / convert_factor[return_units]
