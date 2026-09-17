import unittest

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - exercised on CPU-only CI
    cp = None

from qc_agent.core.dmrg import run_dmrg
from qc_agent.core.mps_runtime import MPSRuntime
from qc_agent.models import TNGate, TNPayload
from qc_agent.plugins.lattice import build_spin_hamiltonian
from qc_agent.plugins.models import (
    DMRGPayload,
    LatticeHamiltonianPayload,
    PauliTerm,
    PEPSPayload,
    TEBDPayload,
)
from qc_agent.core.peps import run_peps
from qc_agent.plugins.tebd import run_tebd


def _cuda_ready() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@unittest.skipUnless(_cuda_ready(), "CUDA GPU is required for CPU/GPU agreement tests")
class GPUAgreementTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def test_mps_observables_agree_with_cpu(self):
        payload = TNPayload(
            n_qubits=4,
            gates=[
                TNGate(name="h", target=0),
                TNGate(name="ry", target=1, theta=0.31),
                TNGate(name="cx", control=0, target=3),
                TNGate(name="rz", target=2, theta=-0.22),
            ],
            dtype="complex64",
            bond_dim=4,
        )
        terms = [
            PauliTerm(paulis={0: "X", 1: "Y"}, coefficient=0.7),
            PauliTerm(paulis={0: "Z", 3: "Z"}, coefficient=-0.2),
            PauliTerm(paulis={2: "X"}, coefficient=0.3),
        ]
        cpu = MPSRuntime(np, payload)
        gpu = MPSRuntime(cp, payload)
        cpu_values = cpu.expectation(terms)
        gpu_values = gpu.expectation(terms)
        self.assertEqual(len(cpu_values), len(gpu_values))
        for actual, expected in zip(gpu_values, cpu_values):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertAlmostEqual(gpu.norm2(), cpu.norm2(), places=5)

    def test_dmrg_agrees_with_cpu_on_heisenberg_chain(self):
        spec = LatticeHamiltonianPayload(dimensions=[4], model="heisenberg", field=0.3)
        terms = [PauliTerm(**term) for term in build_spin_hamiltonian(spec)["terms"]]
        payload = DMRGPayload(
            n_qubits=4,
            terms=terms,
            dtype="complex64",
            bond_dim=4,
            sweeps=4,
            tolerance=1e-5,
            residual_tolerance=1e-4,
            lanczos_maxiter=16,
        )
        cpu = run_dmrg(np, payload)
        gpu = run_dmrg(cp, payload)
        self.assertAlmostEqual(gpu["ground_energy"], cpu["ground_energy"], places=4)
        self.assertAlmostEqual(gpu["norm2"], cpu["norm2"], places=4)
        self.assertTrue(gpu["converged"])
        self.assertIn("energy_variance", gpu)
        self.assertIn("local_solver_residual", gpu)

    def test_tebd_agrees_with_cpu_and_reports_diagnostics(self):
        payload = TEBDPayload(
            n_qubits=4,
            terms=[
                PauliTerm(paulis={0: "X", 1: "X"}, coefficient=0.4),
                PauliTerm(paulis={1: "Z", 2: "Z"}, coefficient=0.3),
                PauliTerm(paulis={2: "Y", 3: "Y"}, coefficient=-0.2),
            ],
            dt=0.03,
            steps=3,
            bond_dim=4,
        )
        cpu = run_tebd(np, payload)
        gpu = run_tebd(cp, payload)
        self.assertEqual(len(gpu["energies"]), len(cpu["energies"]))
        for actual, expected in zip(gpu["energies"], cpu["energies"]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertAlmostEqual(gpu["norm2"], cpu["norm2"], places=4)
        self.assertLess(gpu["norm_drift"], 1e-4)
        self.assertGreaterEqual(gpu["bond_growth"], 0)
        self.assertEqual(len(gpu["discarded_weight_history"]), payload.steps + 1)
        self.assertIn("truncation_diagnostics", gpu)

    def test_boundary_mps_peps_agrees_with_cpu_and_reports_environment(self):
        payload = PEPSPayload(
            n_qubits=9,
            lattice={"dimensions": [3, 3], "boundary": "open"},
            terms=[
                PauliTerm(paulis={0: "X", 1: "X"}, coefficient=0.2),
                PauliTerm(paulis={1: "Z", 4: "Z"}, coefficient=0.3),
            ],
            dtype="complex64",
            bond_dim=2,
            boundary_bond_dim=4,
            contraction_method="boundary-mps",
            dt=0.02,
            steps=1,
        )
        cpu = run_peps(np, payload)
        gpu = run_peps(cp, payload)
        self.assertEqual(gpu["contraction_method"], "boundary-mps")
        self.assertEqual(len(gpu["boundary_diagnostics"]["rows"]), 3)
        self.assertAlmostEqual(gpu["norm2"], cpu["norm2"], places=4)
        self.assertAlmostEqual(gpu["energies"][-1], cpu["energies"][-1], places=4)
        self.assertLess(abs(gpu["norm2"] - 1.0), 1e-4)

    def test_2d_spin_models_run_with_bounded_boundary_mps(self):
        for model, anisotropy in (("ising", 1.0), ("heisenberg", 1.0)):
            with self.subTest(model=model):
                spec = LatticeHamiltonianPayload(
                    dimensions=[3, 3],
                    model=model,
                    coupling=1.0,
                    field=0.3,
                    anisotropy=anisotropy,
                )
                terms = [PauliTerm(**term) for term in build_spin_hamiltonian(spec)["terms"]]
                payload = PEPSPayload(
                    n_qubits=9,
                    lattice={"dimensions": [3, 3], "boundary": "open"},
                    terms=terms,
                    dtype="complex64",
                    bond_dim=2,
                    boundary_bond_dim=4,
                    contraction_method="boundary-mps",
                    dt=0.01,
                    steps=1,
                )
                result = run_peps(cp, payload)
                self.assertEqual(result["contraction_method"], "boundary-mps")
                self.assertEqual(len(result["boundary_diagnostics"]["rows"]), 3)
                self.assertTrue(np.isfinite(result["energies"][-1]))
                self.assertTrue(np.isfinite(result["norm2"]))
                self.assertFalse(result["resource_estimate"].get("materializes_statevector", False))


if __name__ == "__main__":
    unittest.main()
