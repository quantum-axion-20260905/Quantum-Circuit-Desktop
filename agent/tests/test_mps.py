import math
import unittest

import numpy as np

from qc_agent.backends.mps import amplitudes, sample
from qc_agent.models import RunPayload, TNGate, TNPayload


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


if __name__ == "__main__":
    unittest.main()
