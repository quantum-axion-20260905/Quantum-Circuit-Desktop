import unittest
from types import SimpleNamespace

from qc_agent.backends.preflight import estimate
from qc_agent.models import PreflightPayload


class PreflightTests(unittest.TestCase):
    def test_rejects_same_control_and_target(self):
        payload = SimpleNamespace(
            n_qubits=2,
            gates=[SimpleNamespace(name="cx", control=1, target=1)],
            dtype="complex64",
            optimize="auto",
            max_mem_mb=4096,
            max_time_ms=120000,
        )
        result = estimate(payload)
        self.assertFalse(result["feasible"])
        self.assertIn("control and target", result["warnings"][0])

    def test_large_mps_preflight_keeps_statevector_bound_representable(self):
        result = estimate(PreflightPayload(n_qubits=4096, bond_dim=2, shots=1))
        self.assertTrue(result["feasible"])
        self.assertEqual(result["estimate_method"], "mps-bond-dimension-bound")
        self.assertEqual(result["bond_dim"], 2)
        self.assertLess(result["estimated_peak_memory_mb"], 1)

    def test_noise_selected_amplitude_is_rejected_before_execution(self):
        from qc_agent.models import NoiseModel

        result = estimate(PreflightPayload(
            n_qubits=2,
            result_type="selected_amplitudes",
            noise=NoiseModel(readout_flip=0.1),
        ))
        self.assertFalse(result["feasible"])
        self.assertIn("result_type=samples", result["warnings"][0])

    def test_does_not_cap_memory_estimate_at_twenty_qubits(self):
        result = estimate(PreflightPayload(n_qubits=64, backend="statevector"), backend_name="statevector")
        self.assertFalse(result["feasible"])
        self.assertGreater(result["estimated_peak_memory_mb"], 4096)


if __name__ == "__main__":
    unittest.main()
