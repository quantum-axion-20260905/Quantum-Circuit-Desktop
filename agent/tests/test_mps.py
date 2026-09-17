import math
import hashlib
import tempfile
import unittest

import numpy as np

from qc_agent.backends.mps import amplitudes, sample
from qc_agent.core.contracts import CheckpointManifest
from qc_agent.core.mps_runtime import MPSRuntime
from qc_agent.models import RunPayload, TNGate, TNPayload
from qc_agent.plugins.models import PauliTerm


class MPSSimulatorTests(unittest.TestCase):
    def test_bell_state_amplitudes_and_sampling(self):
        gates = [
            TNGate(name="h", target=0),
            TNGate(name="cx", control=0, target=1),
        ]
        result = amplitudes(
            np,
            TNPayload(n_qubits=2, gates=gates, bitstrings=["00", "11"], bond_dim=2),
        )
        self.assertEqual(result["backend"], "tensor-network-mps")
        self.assertAlmostEqual(result["norm2"], 1.0, places=5)
        self.assertEqual(result["bond_dim_used"], 2)
        self.assertAlmostEqual(result["amplitudes"][0]["re"], 1 / math.sqrt(2), places=5)
        self.assertAlmostEqual(result["amplitudes"][1]["re"], 1 / math.sqrt(2), places=5)

        sampled = sample(
            np,
            RunPayload(n_qubits=2, gates=gates, shots=128, seed=7, bond_dim=2),
        )
        self.assertEqual(sampled["backend"], "tensor-network-mps-sample")
        self.assertEqual(sum(sampled["counts"].values()), 128)
        self.assertEqual(set(sampled["counts"]), {"00", "11"})
        self.assertTrue(sampled["validation"]["normalization_passed"])

    def test_reversed_non_adjacent_control_and_target(self):
        result = amplitudes(
            np,
            TNPayload(
                n_qubits=4,
                gates=[
                    TNGate(name="h", target=3),
                    TNGate(name="cx", control=3, target=0),
                ],
                bitstrings=["0000", "1001", "0001", "1000"],
                bond_dim=2,
            ),
        )
        values = {item["bitstring"]: complex(item["re"], item["im"]) for item in result["amplitudes"]}
        self.assertAlmostEqual(abs(values["0000"]), 1 / math.sqrt(2), places=5)
        self.assertAlmostEqual(abs(values["1001"]), 1 / math.sqrt(2), places=5)
        self.assertAlmostEqual(abs(values["0001"]), 0.0, places=5)
        self.assertAlmostEqual(abs(values["1000"]), 0.0, places=5)
        self.assertAlmostEqual(result["norm2"], 1.0, places=5)

    def test_low_bond_run_exposes_truncation_diagnostics(self):
        result = amplitudes(
            np,
            TNPayload(
                n_qubits=2,
                gates=[TNGate(name="h", target=0), TNGate(name="cx", control=0, target=1)],
                bitstrings=["00", "11"],
                bond_dim=1,
            ),
        )
        self.assertTrue(result["approximate"])
        self.assertGreater(result["discarded_weight"], 0.49)
        self.assertLess(result["norm2"], 0.51)
        self.assertTrue(result["warnings"])

    def test_64_qubit_ghz_with_bond_two(self):
        gates = [TNGate(name="h", target=0)] + [
            TNGate(name="cx", control=index, target=index + 1)
            for index in range(63)
        ]
        result = amplitudes(
            np,
            TNPayload(
                n_qubits=64,
                gates=gates,
                bitstrings=["0" * 64, "1" * 64],
                bond_dim=2,
            ),
        )
        self.assertLessEqual(result["bond_dim_used"], 2)
        self.assertAlmostEqual(result["norm2"], 1.0, places=4)
        for item in result["amplitudes"]:
            self.assertAlmostEqual(abs(complex(item["re"], item["im"])), 1 / math.sqrt(2), places=4)

    def test_256_qubit_low_bond_sampling_does_not_need_statevector(self):
        result = sample(
            np,
            RunPayload(n_qubits=256, shots=8, seed=11, bond_dim=2),
        )
        self.assertEqual(result["n_qubits"], 256)
        self.assertEqual(result["counts"], {"0" * 256: 8})
        self.assertEqual(result["bond_dim_used"], 1)
        self.assertAlmostEqual(result["norm2"], 1.0, places=5)

    def test_canonicalization_preserves_norm_and_bell_correlations(self):
        runtime = MPSRuntime(
            np,
            TNPayload(
                n_qubits=2,
                gates=[TNGate(name="h", target=0), TNGate(name="cx", control=0, target=1)],
                bond_dim=2,
            ),
        )
        terms = [PauliTerm(paulis={0: "Z", 1: "Z"}, coefficient=1.0)]
        before = runtime.norm2()
        runtime.canonicalize_left()
        middle = runtime.norm2()
        runtime.canonicalize_right()
        after = runtime.norm2()
        self.assertAlmostEqual(before, middle, places=5)
        self.assertAlmostEqual(middle, after, places=5)
        self.assertAlmostEqual(runtime.expectation(terms)[0], 1.0, places=5)

    def test_energy_moments_report_variance(self):
        runtime = MPSRuntime(np, TNPayload(n_qubits=1, gates=[], bond_dim=2))
        energy, second_moment, variance = runtime.energy_moments([
            PauliTerm(paulis={0: "X"}, coefficient=1.0),
        ])
        self.assertAlmostEqual(energy, 0.0, places=6)
        self.assertAlmostEqual(second_moment, 1.0, places=6)
        self.assertAlmostEqual(variance, 1.0, places=6)

    def test_mps_resource_estimate_is_non_allocating_and_conservative(self):
        runtime = MPSRuntime(np, TNPayload(n_qubits=8, gates=[], bond_dim=4))
        estimate = runtime.estimate_resources()
        self.assertEqual(estimate["representation"], "mps")
        self.assertGreater(estimate["tensor_bytes"], 0)
        self.assertGreaterEqual(estimate["peak_bytes_estimate"], estimate["tensor_bytes"])

    def test_checkpoint_round_trip_restores_mps_state_atomically(self):
        runtime = MPSRuntime(
            np,
            TNPayload(
                n_qubits=2,
                gates=[TNGate(name="h", target=0), TNGate(name="cx", control=0, target=1)],
                bond_dim=2,
            ),
        )
        manifest = CheckpointManifest(
            checkpoint_id="mps-test",
            request_sha256=hashlib.sha256(b"mps-test").hexdigest(),
            method="finite-two-site-dmrg",
            representation="mps",
            dtype="complex64",
            device="cpu",
            step=2,
            created_at="2026-09-17T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/state.npz"
            saved = runtime.save_checkpoint(path, manifest)
            self.assertEqual(saved["schema"], "quantum-circuit/checkpoint-v1")
            restored = MPSRuntime(np, TNPayload(n_qubits=2, gates=[], bond_dim=2))
            loaded = restored.restore_checkpoint(path)
            self.assertEqual(loaded["checkpoint_id"], "mps-test")
            self.assertAlmostEqual(restored.norm2(), runtime.norm2(), places=5)
            self.assertAlmostEqual(
                restored.expectation([PauliTerm(paulis={0: "Z", 1: "Z"}, coefficient=1.0)])[0],
                1.0,
                places=5,
            )


if __name__ == "__main__":
    unittest.main()
