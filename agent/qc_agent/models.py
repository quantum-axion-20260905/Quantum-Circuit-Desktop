from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Gate(BaseModel):
    name: Literal["h", "x", "rx", "ry", "rz"]
    target: int = Field(ge=0)
    theta: float | None = None


class TNGate(BaseModel):
    name: Literal["h", "x", "rx", "ry", "rz", "cx", "cz"]
    target: int = Field(ge=0)
    control: int | None = Field(default=None, ge=0)
    theta: float | None = None


class SimulatePayload(BaseModel):
    n_qubits: int = Field(ge=1, le=30)
    gates: list[Gate] = Field(default_factory=list)
    dtype: Literal["complex64", "complex128"] = "complex64"


class SamplePayload(BaseModel):
    n_qubits: int = Field(ge=1, le=20)
    gates: list[TNGate] = Field(default_factory=list)
    shots: int = Field(default=1024, ge=1, le=200000)
    dtype: Literal["complex64", "complex128"] = "complex64"


class BenchPayload(BaseModel):
    size: int = Field(default=2048, ge=256, le=8192)
    iters: int = Field(default=10, ge=1, le=200)
    dtype: Literal["fp16", "fp32"] = "fp16"


class TNPayload(BaseModel):
    n_qubits: int = Field(ge=1, le=64)
    gates: list[TNGate] = Field(default_factory=list)
    bitstrings: list[str] = Field(default_factory=list, max_length=64)
    dtype: Literal["complex64", "complex128"] = "complex64"
    optimize: Literal["auto", "cotengra"] = "auto"


class RunPayload(TNPayload):
    backend: Literal["auto", "reference", "tensor-network"] = "auto"
    shots: int = Field(default=1024, ge=1, le=200000)
    seed: int | None = None
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)
    result_type: Literal[
        "selected_amplitudes", "samples", "full_state", "expectation_value"
    ] = "selected_amplitudes"


class PreflightPayload(TNPayload):
    backend: Literal["auto", "reference", "tensor-network"] = "auto"
    max_time_ms: int = Field(default=120000, ge=100, le=3600000)
    max_mem_mb: float = Field(default=4096, gt=0, le=1048576)
