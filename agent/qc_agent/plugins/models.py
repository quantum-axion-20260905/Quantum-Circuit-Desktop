from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..backends.limits import MAX_EXACT_DIAGONALIZATION_QUBITS, MAX_REFERENCE_QUBITS
from ..models import TNGate


Pauli = Literal["I", "X", "Y", "Z"]
FermionAction = Literal["create", "annihilate"]


class LatticeSpec(BaseModel):
    """Rectangular 1D/2D/3D lattice with deterministic snake ordering."""

    dimensions: list[int] = Field(min_length=1, max_length=3)
    boundary: Literal["open", "periodic"] = "open"

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: list[int]) -> list[int]:
        if any(size < 1 or size > 64 for size in value):
            raise ValueError("each lattice dimension must be between 1 and 64")
        sites = math.prod(value)
        if sites > 4096:
            raise ValueError("lattice is limited to 4096 sites")
        return value

    @property
    def n_sites(self) -> int:
        return math.prod(self.dimensions)


class PauliTerm(BaseModel):
    """Sparse Hermitian Pauli term; omitted qubits are identity operators."""

    paulis: dict[int, Pauli] = Field(default_factory=dict)
    coefficient: float
    label: str | None = None

    @field_validator("coefficient")
    @classmethod
    def finite_coefficient(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("coefficient must be finite")
        return value

    @field_validator("paulis")
    @classmethod
    def valid_indices(cls, value: dict[int, Pauli]) -> dict[int, Pauli]:
        if any(index < 0 for index in value):
            raise ValueError("Pauli indices must be non-negative")
        return {int(index): pauli for index, pauli in value.items() if pauli != "I"}


class FermionOperator(BaseModel):
    mode: int = Field(ge=0)
    action: FermionAction


class FermionTerm(BaseModel):
    """Ordered product of fermionic creation/annihilation operators."""

    operators: list[FermionOperator] = Field(min_length=1, max_length=8)
    coefficient: float
    label: str | None = None

    @field_validator("coefficient")
    @classmethod
    def finite_coefficient(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fermionic coefficient must be finite")
        return value


class FermionMappingPayload(BaseModel):
    n_modes: int = Field(ge=1, le=4096)
    terms: list[FermionTerm] = Field(min_length=1, max_length=8192)
    mapping: Literal["jordan_wigner"] = "jordan_wigner"
    hermitian_tolerance: float = Field(default=1e-10, gt=0, le=1e-3)

    @model_validator(mode="after")
    def validate_modes(self):
        for term in self.terms:
            if any(operator.mode >= self.n_modes for operator in term.operators):
                raise ValueError("fermion operator mode exceeds n_modes")
        return self


class ExpectationPayload(BaseModel):
    n_qubits: int = Field(ge=1, le=4096)
    gates: list[TNGate] = Field(default_factory=list)
    terms: list[PauliTerm] = Field(min_length=1, max_length=8192)
    dtype: Literal["complex64", "complex128"] = "complex64"
    backend: Literal["auto", "reference", "statevector", "tensor-network"] = "auto"
    bond_dim: int = Field(default=64, ge=1, le=4096)
    truncation_cutoff: float = Field(default=0.0, ge=0.0, le=1.0)
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)

    @model_validator(mode="after")
    def validate_terms(self):
        for term in self.terms:
            if any(index >= self.n_qubits for index in term.paulis):
                raise ValueError("Pauli term index exceeds n_qubits")
        return self


class ObservableCrossValidatePayload(ExpectationPayload):
    """Small-system MPS observable validation against the CPU reference."""

    tolerance: float = Field(default=1e-5, gt=0, le=1.0)

    @model_validator(mode="after")
    def validate_reference_size(self):
        if self.n_qubits > MAX_REFERENCE_QUBITS:
            raise ValueError(f"observable cross-validation supports at most {MAX_REFERENCE_QUBITS} qubits")
        return self


class GroundStatePayload(BaseModel):
    """Small exact diagonalization reference for Hamiltonian validation."""

    n_qubits: int = Field(ge=1, le=MAX_EXACT_DIAGONALIZATION_QUBITS)
    terms: list[PauliTerm] = Field(min_length=1, max_length=8192)
    bitstrings: list[str] = Field(default_factory=list, max_length=64)
    dtype: Literal["complex64", "complex128"] = "complex64"
    backend: Literal["auto", "exact-diagonalization"] = "auto"
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)

    @model_validator(mode="after")
    def validate_terms(self):
        if any(index >= self.n_qubits for term in self.terms for index in term.paulis):
            raise ValueError("Pauli term index exceeds n_qubits")
        for raw in self.bitstrings:
            value = raw.strip().replace("_", "")
            if len(value) != self.n_qubits or any(char not in "01" for char in value):
                raise ValueError(f"invalid bitstring for {self.n_qubits} qubits: {raw!r}")
        return self


class DMRGPayload(BaseModel):
    """Finite-size two-site DMRG over a sparse Pauli Hamiltonian."""

    n_qubits: int = Field(ge=2, le=4096)
    terms: list[PauliTerm] = Field(min_length=1, max_length=8192)
    observables: list[PauliTerm] = Field(default_factory=list, max_length=1024)
    initial_bitstring: str = ""
    dtype: Literal["complex64", "complex128"] = "complex64"
    backend: Literal["auto", "tensor-network"] = "auto"
    bond_dim: int = Field(default=16, ge=1, le=64)
    truncation_cutoff: float = Field(default=0.0, ge=0.0, le=1.0)
    sweeps: int = Field(default=4, ge=1, le=64)
    tolerance: float = Field(default=1e-7, gt=0, le=1.0)
    residual_tolerance: float = Field(default=1e-6, gt=0, le=1.0)
    variance_tolerance: float | None = Field(default=None, gt=0, le=1.0)
    local_solver: Literal["lanczos", "dense"] = "lanczos"
    lanczos_maxiter: int = Field(default=32, ge=4, le=128)
    lanczos_tolerance: float = Field(default=1e-8, gt=0, le=1e-2)
    checkpoint_path: str | None = Field(default=None, min_length=1, max_length=4096)
    resume_from: str | None = Field(default=None, min_length=1, max_length=4096)
    max_term_locality: int = Field(default=64, ge=1, le=4096)
    max_local_dim: int = Field(default=4096, ge=16, le=16384)
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)

    @model_validator(mode="after")
    def validate_payload(self):
        value = self.initial_bitstring.strip().replace("_", "") or ("0" * self.n_qubits)
        if len(value) != self.n_qubits or any(char not in "01" for char in value):
            raise ValueError(f"invalid initial_bitstring for {self.n_qubits} qubits")
        self.initial_bitstring = value
        for term in [*self.terms, *self.observables]:
            if any(index >= self.n_qubits for index in term.paulis):
                raise ValueError("Pauli term index exceeds n_qubits")
        for term in self.terms:
            if len(term.paulis) > self.max_term_locality:
                raise ValueError(f"DMRG term locality exceeds max_term_locality={self.max_term_locality}")
        if 4 * self.bond_dim * self.bond_dim > self.max_local_dim:
            raise ValueError("bond_dim produces a two-site local problem larger than max_local_dim")
        return self


class IPEPSInteraction(BaseModel):
    """A translationally repeated two-site interaction in an iPEPS unit cell."""

    left_site: int = Field(ge=0)
    right_site: int = Field(ge=0)
    displacement: list[int] = Field(min_length=2, max_length=2)
    left_pauli: Pauli
    right_pauli: Pauli
    coefficient: float
    label: str | None = None

    @field_validator("coefficient")
    @classmethod
    def finite_coefficient(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("iPEPS interaction coefficient must be finite")
        return value

    @field_validator("displacement")
    @classmethod
    def finite_displacement(cls, value: list[int]) -> list[int]:
        if value[0] == 0 and value[1] == 0:
            raise ValueError("iPEPS interaction displacement must be non-zero")
        return [int(value[0]), int(value[1])]


class CTMRGPayload(BaseModel):
    """Bounded infinite-2D iPEPS/CTMRG problem contract.

    The unit cell and repeated interactions are explicit.  This prevents a
    future solver from silently interpreting a finite lattice Hamiltonian as
    an infinite system.  The first implementation supports small unit cells;
    larger cells must pass a new admission contract before being enabled.
    """

    unit_cell: list[int] = Field(default_factory=lambda: [1, 1], min_length=2, max_length=2)
    terms: list[PauliTerm] = Field(default_factory=list, max_length=1024)
    interactions: list[IPEPSInteraction] = Field(min_length=1, max_length=1024)
    dtype: Literal["complex64", "complex128"] = "complex64"
    backend: Literal["auto", "tensor-network"] = "auto"
    physical_bond_dim: int = Field(default=2, ge=1, le=8)
    virtual_bond_dim: int = Field(default=1, ge=1, le=8)
    tensor_data: list[list[float]] | None = Field(default=None, max_length=32768)
    environment_bond_dim: int = Field(default=16, ge=1, le=128)
    iterations: int = Field(default=20, ge=1, le=200)
    tolerance: float = Field(default=1e-8, gt=0, le=1.0)
    optimization: Literal["none", "product-coordinate-descent", "simple-update", "full-update"] = "none"
    optimization_steps: int = Field(default=32, ge=1, le=256)
    optimization_tolerance: float = Field(default=1e-7, gt=0, le=1.0)
    optimization_dt: float = Field(default=0.01, gt=0, le=1.0)
    full_update_step: float = Field(default=0.05, gt=0, le=1.0)
    full_update_max_parameters: int = Field(default=128, ge=1, le=256)
    full_update_optimizer: Literal["coordinate", "finite-difference-gradient", "spsa-gradient", "finite-torus-gradient", "autodiff-ctmrg-gradient", "implicit-ctmrg-gradient"] = "coordinate"
    full_update_gradient_epsilon: float = Field(default=1e-3, gt=0, le=0.1)
    full_update_spsa_directions: int = Field(default=4, ge=1, le=16)
    full_update_max_evaluations: int = Field(default=1024, ge=8, le=4096)
    full_update_implicit_iterations: int = Field(default=32, ge=1, le=256)
    full_update_implicit_tolerance: float = Field(default=1e-6, gt=0, le=1.0)
    full_update_implicit_damping: float = Field(default=0.5, gt=0, le=1.0)
    gauge_validation: bool = False
    gauge_validation_tolerance: float = Field(default=1e-4, gt=0, le=1.0)
    initial_state: Literal["up", "down", "plus", "neel"] = "up"
    checkpoint_path: str | None = Field(default=None, min_length=1, max_length=4096)
    resume_from: str | None = Field(default=None, min_length=1, max_length=4096)
    optimizer_checkpoint_path: str | None = Field(default=None, min_length=1, max_length=4096)
    optimizer_resume_from: str | None = Field(default=None, min_length=1, max_length=4096)
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)

    @field_validator("unit_cell")
    @classmethod
    def validate_unit_cell(cls, value: list[int]) -> list[int]:
        if any(size < 1 or size > 2 for size in value):
            raise ValueError("CTMRG unit-cell dimensions must be between 1 and 2")
        if math.prod(value) > 4:
            raise ValueError("CTMRG currently supports at most a 2x2 unit cell")
        return [int(size) for size in value]

    @field_validator("tensor_data")
    @classmethod
    def validate_tensor_data(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return None
        for index, pair in enumerate(value):
            if len(pair) != 2:
                raise ValueError(f"iPEPS tensor_data[{index}] must be a [real, imaginary] pair")
            if not all(math.isfinite(float(component)) for component in pair):
                raise ValueError(f"iPEPS tensor_data[{index}] must contain finite numbers")
        return [[float(pair[0]), float(pair[1])] for pair in value]

    @model_validator(mode="after")
    def validate_payload(self):
        cell_sites = math.prod(self.unit_cell)
        if self.optimization != "none" and (self.checkpoint_path is not None or self.resume_from is not None):
            raise ValueError(
                "CTMRG optimizer checkpoint/resume is not supported yet; checkpointing currently covers contraction-only runs"
            )
        if (self.optimizer_checkpoint_path is not None or self.optimizer_resume_from is not None) and (
            self.optimization != "full-update"
            or self.full_update_optimizer not in {
                "coordinate",
                "finite-difference-gradient",
                "finite-torus-gradient",
                "spsa-gradient",
            }
        ):
            raise ValueError(
                "optimizer checkpoint state requires a supported full-update optimizer strategy"
            )
        expected_tensor_values = (
            cell_sites * int(self.physical_bond_dim) * int(self.virtual_bond_dim) ** 4
        )
        if self.tensor_data is not None and len(self.tensor_data) != expected_tensor_values:
            raise ValueError(
                "iPEPS tensor_data length must equal unit_cell_sites * physical_bond_dim * virtual_bond_dim**4 "
                f"({expected_tensor_values})"
            )
        for term in self.terms:
            if any(index >= cell_sites for index in term.paulis):
                raise ValueError("iPEPS onsite term index exceeds the unit-cell site count")
        for interaction in self.interactions:
            if interaction.left_site >= cell_sites or interaction.right_site >= cell_sites:
                raise ValueError("iPEPS interaction site exceeds the unit-cell site count")
        return self


class CTMRGConvergenceStudyPayload(BaseModel):
    """Explicit API contract for independent environment-chi CTMRG points."""

    problem: CTMRGPayload
    environment_bond_dims: list[int] = Field(min_length=1, max_length=8)
    backend: Literal["auto", "tensor-network"] = "auto"
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)

    @field_validator("environment_bond_dims")
    @classmethod
    def validate_environment_bond_dims(cls, value: list[int]) -> list[int]:
        normalized = [int(item) for item in value]
        if any(item < 1 or item > 128 for item in normalized):
            raise ValueError("environment_bond_dims values must be between 1 and 128")
        if len(set(normalized)) != len(normalized):
            raise ValueError("environment_bond_dims values must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_problem(self):
        if self.problem.optimization != "none":
            raise ValueError("CTMRG convergence studies require problem.optimization='none'")
        return self


class PEPSPayload(BaseModel):
    """Finite 2D/3D PEPS simple-update evolution with bounded contraction.

    Norms and local observables use a double-layer contraction, so the
    payload is not limited by a materialized ``2**n`` statevector.  The
    admission controller still bounds lattice size, bond dimension and
    contraction width before work is scheduled.
    """

    n_qubits: int = Field(ge=1, le=64)
    terms: list[PauliTerm] = Field(min_length=1, max_length=8192)
    observables: list[PauliTerm] = Field(default_factory=list, max_length=1024)
    lattice: LatticeSpec
    dtype: Literal["complex64", "complex128"] = "complex64"
    backend: Literal["auto", "tensor-network"] = "auto"
    bond_dim: int = Field(default=2, ge=1, le=4)
    truncation_cutoff: float = Field(default=0.0, ge=0.0, le=1.0)
    dt: float = Field(default=0.01)
    steps: int = Field(default=1, ge=1, le=64)
    order: Literal[1, 2] = 2
    contraction_method: Literal["auto", "opt_einsum", "enumeration", "boundary-mps"] = "auto"
    boundary_bond_dim: int = Field(default=32, ge=1, le=4096)
    boundary_checkpoint_path: str | None = Field(default=None, min_length=1, max_length=4096)
    boundary_resume_from: str | None = Field(default=None, min_length=1, max_length=4096)
    max_contraction_states: int = Field(default=1_000_000, ge=1, le=10_000_000)
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)

    @field_validator("dt")
    @classmethod
    def finite_dt(cls, value: float) -> float:
        if not math.isfinite(value) or value == 0:
            raise ValueError("dt must be finite and non-zero")
        return value

    @model_validator(mode="after")
    def validate_payload(self):
        if len(self.lattice.dimensions) < 2:
            raise ValueError("native PEPS requires a 2D or 3D lattice")
        if self.lattice.n_sites != self.n_qubits:
            raise ValueError("PEPS lattice site count must equal n_qubits")
        from .lattice import lattice_graph

        edge_set = {
            frozenset((int(edge["source"]), int(edge["target"])))
            for edge in lattice_graph(self.lattice)["edges"]
        }
        for term in [*self.terms, *self.observables]:
            if any(index >= self.n_qubits for index in term.paulis):
                raise ValueError("Pauli term index exceeds n_qubits")
        if any(len(term.paulis) > 2 for term in self.terms):
            raise ValueError("native PEPS currently supports one- and two-site Pauli terms")
        if any(
            len(term.paulis) == 2 and frozenset(int(index) for index in term.paulis) not in edge_set
            for term in self.terms
        ):
            raise ValueError("native PEPS two-site evolution terms must follow a lattice edge")
        return self


class LatticeHamiltonianPayload(LatticeSpec):
    model: Literal["ising", "heisenberg", "xxz"] = "ising"
    coupling: float = 1.0
    field: float = 0.0
    anisotropy: float = 1.0

    @field_validator("coupling", "field", "anisotropy")
    @classmethod
    def finite_parameter(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Hamiltonian parameters must be finite")
        return value


class CTMRGSpinModelPayload(LatticeHamiltonianPayload):
    """Spin-model builder contract for periodic iPEPS unit cells."""

    initial_state: Literal["up", "down", "plus", "neel"] = "up"
    environment_bond_dim: int = Field(default=16, ge=1, le=128)
    iterations: int = Field(default=20, ge=1, le=200)
    tolerance: float = Field(default=1e-8, gt=0, le=1.0)
    gauge_validation: bool = False
    gauge_validation_tolerance: float = Field(default=1e-4, gt=0, le=1.0)

    @model_validator(mode="after")
    def validate_ctmrg_geometry(self):
        if len(self.dimensions) != 2:
            raise ValueError("CTMRG spin models require a 2D unit-cell dimensions=[nx, ny]")
        if any(int(size) > 2 for size in self.dimensions):
            raise ValueError("CTMRG spin model unit-cell dimensions are limited to 2x2")
        return self


class HubbardPayload(LatticeSpec):
    """Spinful Hubbard model on a rectangular lattice.

    Each site contributes two fermionic modes (up/down), so the mapped qubit
    count is twice the lattice site count.
    """

    hopping: float = 1.0
    onsite_u: float = 0.0
    chemical_potential: float = 0.0

    @field_validator("hopping", "onsite_u", "chemical_potential")
    @classmethod
    def finite_parameter(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Hubbard parameters must be finite")
        return value

    @model_validator(mode="after")
    def validate_mapped_size(self):
        if self.n_sites * 2 > 4096:
            raise ValueError("spinful Hubbard mapping is limited to 4096 qubits")
        return self


class TEBDPayload(BaseModel):
    n_qubits: int = Field(ge=1, le=4096)
    gates: list[TNGate] = Field(default_factory=list)
    terms: list[PauliTerm] = Field(min_length=1, max_length=8192)
    observables: list[PauliTerm] = Field(default_factory=list, max_length=1024)
    lattice: LatticeSpec | None = None
    dtype: Literal["complex64", "complex128"] = "complex64"
    bond_dim: int = Field(default=64, ge=1, le=4096)
    truncation_cutoff: float = Field(default=0.0, ge=0.0, le=1.0)
    dt: float = Field(default=0.01)
    steps: int = Field(default=10, ge=1, le=10000)
    order: Literal[1, 2] = 2
    max_term_locality: int = Field(default=64, ge=1, le=4096)
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)

    @field_validator("dt")
    @classmethod
    def finite_dt(cls, value: float) -> float:
        if not math.isfinite(value) or value == 0:
            raise ValueError("dt must be finite and non-zero")
        return value

    @model_validator(mode="after")
    def validate_terms(self):
        if self.lattice is not None and self.lattice.n_sites != self.n_qubits:
            raise ValueError("lattice site count must equal n_qubits")
        for term in [*self.terms, *self.observables]:
            if any(index >= self.n_qubits for index in term.paulis):
                raise ValueError("Pauli term index exceeds n_qubits")
        for term in self.terms:
            if len(term.paulis) > self.max_term_locality:
                raise ValueError(f"TEBD term locality exceeds max_term_locality={self.max_term_locality}")
        return self
