import unittest

from qc_agent.backends.reference import run
from qc_agent.models import RunPayload, TNGate


class ReferenceBackendTests(unittest.TestCase):
    def test_bell_state(self):
        out = run(RunPayload(n_qubits=2, backend="reference", gates=[
            TNGate(name="h", target=0), TNGate(name="cx", control=0, target=1)
        ], shots=256, seed=7, bitstrings=["00", "01", "10", "11"]))
        amps = {x["bitstring"]: complex(x["re"], x["im"]) for x in out["amplitudes"]}
        self.assertAlmostEqual(abs(amps["00"]) ** 2, 0.5, places=6)
        self.assertAlmostEqual(abs(amps["11"]) ** 2, 0.5, places=6)
        self.assertAlmostEqual(out["norm2"], 1.0, places=6)
        self.assertEqual(sum(out["counts"].values()), 256)

    def test_seed_is_deterministic(self):
        payload = RunPayload(n_qubits=1, backend="reference", gates=[TNGate(name="h", target=0)], shots=50, seed=42)
        self.assertEqual(run(payload)["counts"], run(payload)["counts"])


if __name__ == "__main__":
    unittest.main()
