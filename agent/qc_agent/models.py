from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _validate_gate_parameters(name: str, theta: float | None, parameter: str | None, control: int | None, target: int) -> None:
    if theta is not None and not math.isfinite(theta):
        raise ValueError("theta must be finite")
    if parameter is not None and (not parameter or not parameter.replace("_", "a").isalnum()):
        raise ValueError("parameter must contain only letters, numbers, and underscores")
    if parameter is not None and not parameter[0].isalpha():
        raise ValueError("parameter must start with a letter")
    if theta is not None and parameter is not None:
        raise ValueError("provide theta or parameter, not both")
    if name in ("rx", "ry", "rz") and theta is None and parameter is None:
        raise ValueError(f"{name} requires theta or parameter")
    if name not in ("rx", "ry", "rz") and parameter is not None:
        raise ValueError(f"{name} does not accept a parameter")
    if name in ("cx", "cz"):
        if control is None:
            raise ValueError(f"{name} requires control")
        if control == target:
            raise ValueError("control and target must differ")


class Gate(BaseModel):
    name: Literal["h", "x", "rx", "ry", "rz"]
    target: int = Field(ge=0)
    theta: float | None = None

    @model_validator(mode="after")
    def validate_parameters(self):
        _validate_gate_parameters(self.name, self.theta, None, None, self.target)
        return self


class TNGate(BaseModel):
    name: Literal["h", "x", "rx", "ry", "rz", "cx", "cz"]
    target: int = Field(ge=0)
    control: int | None = Field(default=None, ge=0)
    theta: float | None = None
    parameter: str | None = None

    @model_validator(mode="after")
    def validate_parameters(self):
        _validate_gate_parameters(self.name, self.theta, self.parameter, self.control, self.target)
        return self


class NoiseModel(BaseModel):
    """Shot-based Pauli depolarizing noise model.

    The two-qubit channel chooses uniformly from the 15 non-identity Pauli
    products.  Readout flip is applied independently to every measured bit.
    """

    one_qubit_depolarizing: float = Field(default=0.0, ge=0.0, le=1.0)
    two_qubit_depolarizing: float = Field(default=0.0, ge=0.0, le=1.0)
    readout_flip: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def active(self) -> bool:
        return any((self.one_qubit_depolarizing, self.two_qubit_depolarizing, self.readout_flip))


class SimulatePayload(BaseModel):
    n_qubits: int = Field(ge=1, le=20)
    gates: list[TNGate] = Field(default_factory=list)
    dtype: Literal["complex64", "complex128"] = "complex64"
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)


class SamplePayload(BaseModel):
    n_qubits: int = Field(ge=1, le=20)
    gates: list[TNGate] = Field(default_factory=list)
    shots: int = Field(default=1024, ge=1, le=200000)
    dtype: Literal["complex64", "complex128"] = "complex64"
    seed: int | None = None
    noise: NoiseModel | None = None


class BenchPayload(BaseModel):
    size: int = Field(default=2048, ge=256, le=8192)
    iters: int = Field(default=10, ge=1, le=200)
    dtype: Literal["fp16", "fp32"] = "fp16"


class TNPayload(BaseModel):
    n_qubits: int = Field(ge=1, le=4096)
    gates: list[TNGate] = Field(default_factory=list)
    bitstrings: list[str] = Field(default_factory=list, max_length=64)
    dtype: Literal["complex64", "complex128"] = "complex64"
    optimize: Literal["auto", "cotengra"] = "auto"
    tn_method: Literal["mps", "contraction"] = "mps"
    bond_dim: int = Field(default=64, ge=1, le=4096)
    truncation_cutoff: float = Field(default=0.0, ge=0.0, le=1.0)
    noise: NoiseModel | None = None


class RunPayload(TNPayload):
    backend: Literal["auto", "reference", "statevector", "tensor-network"] = "auto"
    shots: int = Field(default=1024, ge=1, le=200000)
    seed: int | None = None
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)
    result_type: Literal["selected_amplitudes", "samples"] = "samples"


class SweepPayload(RunPayload):
    """Finite parameter sweep evaluated sequentially on one selected backend."""

    parameter_values: dict[str, list[float]] = Field(min_length=1)
    parallel_devices: int = Field(default=1, ge=1, le=16)

    @model_validator(mode="after")
    def validate_sweep(self):
        if any(not values for values in self.parameter_values.values()):
            raise ValueError("every sweep parameter needs at least one value")
        if any(not math.isfinite(value) for values in self.parameter_values.values() for value in values):
            raise ValueError("sweep values must be finite")
        points = math.prod(len(values) for values in self.parameter_values.values())
        if points > 256:
            raise ValueError("parameter sweep is limited to 256 points per request")
        parameters = {gate.parameter for gate in self.gates if gate.parameter is not None}
        unknown = set(self.parameter_values) - parameters
        missing = parameters - set(self.parameter_values)
        if unknown:
            raise ValueError(f"sweep contains unknown parameters: {sorted(unknown)}")
        if missing:
            raise ValueError(f"sweep is missing parameters: {sorted(missing)}")
        return self


class PreflightPayload(TNPayload):
    backend: Literal["auto", "reference", "statevector", "tensor-network"] = "auto"
    result_type: Literal["selected_amplitudes", "samples"] = "samples"
    shots: int = Field(default=1024, ge=1, le=200000)
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)


class CrossValidatePayload(TNPayload):
    """Small-circuit validation of GPU/TN amplitudes against the reference."""

    n_qubits: int = Field(ge=1, le=12)
    tolerance: float = Field(default=1e-5, gt=0, le=1)
