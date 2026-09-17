import unittest

import numpy as np

from qc_agent.backends.mps import sample as mps_sample
from qc_agent.backends.statevector import sample as statevector_sample
from qc_agent.models import NoiseModel, RunPayload, SamplePayload, SweepPayload, TNGate


class NoiseTests(unittest.TestCase):
    def test_mps_readout_noise_is_reported_and_counts_are_conserved(self):
        result = mps_sample(
            np,
            RunPayload(
                n_qubits=1,
                gates=[TNGate(name="x", target=0)],
                shots=32,
                seed=5,
                bond_dim=2,
                noise=NoiseModel(readout_flip=1.0),
            ),
        )
        self.assertEqual(result["counts"], {"0": 32})
        self.assertEqual(result["method"], "mps-trajectories")
        self.assertEqual(result["noise"]["model"], "pauli-depolarizing-trajectories")
        self.assertEqual(result["validation"]["counts_total"], 32)

    def test_statevector_pauli_noise_keeps_unit_norm(self):
        result = statevector_sample(
            np,
            SamplePayload(
                n_qubits=1,
                gates=[TNGate(name="h", target=0)],
                shots=32,
                seed=5,
                noise=NoiseModel(one_qubit_depolarizing=1.0),
            ),
        )
        self.assertEqual(sum(result["counts"].values()), 32)
        self.assertTrue(result["validation"]["normalization_passed"])
        self.assertEqual(result["method"], "statevector-trajectories")


class SweepModelTests(unittest.TestCase):
    def test_parameterized_rotation_accepts_cartesian_values(self):
        payload = SweepPayload(
            n_qubits=1,
            gates=[TNGate(name="rx", target=0, parameter="theta")],
            parameter_values={"theta": [0.0, 1.0, 2.0]},
            shots=4,
        )
        self.assertEqual(len(payload.parameter_values["theta"]), 3)

    def test_sweep_rejects_unbound_parameter(self):
        with self.assertRaises(ValueError):
            SweepPayload(
                n_qubits=1,
                gates=[TNGate(name="rx", target=0, parameter="theta")],
                parameter_values={"phi": [0.0, 1.0]},
            )

    def test_empty_parameter_name_is_rejected_cleanly(self):
        with self.assertRaises(ValueError):
            TNGate(name="rx", target=0, parameter="")


if __name__ == "__main__":
    unittest.main()
