import unittest
from tempfile import TemporaryDirectory

from qc_agent.backends.registry import resolve_run_backend
from qc_agent.jobs import JobManager
from qc_agent.models import RunPayload
from qc_agent.provenance import circuit_digest, sha256_json, with_provenance


class RegistryTests(unittest.TestCase):
    def test_auto_uses_gpu_statevector_for_samples(self):
        self.assertEqual(
            resolve_run_backend("auto", "samples", gpu_available=True, tensor_network_available=True),
            "statevector",
        )

    def test_tensor_network_supports_mps_sampling(self):
        self.assertEqual(
            resolve_run_backend("tensor-network", "samples", gpu_available=True, tensor_network_available=True),
            "tensor-network",
        )

    def test_explicit_statevector_requires_gpu(self):
        self.assertEqual(
            resolve_run_backend("statevector", "samples", gpu_available=True, tensor_network_available=False),
            "statevector",
        )
        with self.assertRaises(ValueError):
            resolve_run_backend("statevector", "samples", gpu_available=False, tensor_network_available=False)


class ProvenanceTests(unittest.TestCase):
    def test_canonical_hash_is_order_stable(self):
        self.assertEqual(sha256_json({"b": 2, "a": 1}), sha256_json({"a": 1, "b": 2}))

    def test_result_contains_replay_hashes(self):
        payload = RunPayload(n_qubits=1, seed=11)
        result = with_provenance(
            {"status": "done", "backend": "reference-cpu"},
            payload,
            requested_backend="reference",
            resolved_backend="reference",
            device={"gpu": {"available": False}},
            started_at=0.0,
            agent_version="test",
            seed=payload.seed,
        )
        self.assertEqual(result["provenance"]["circuit_sha256"], circuit_digest(payload))
        self.assertEqual(result["provenance"]["seed"], 11)
        self.assertEqual(len(result["provenance"]["request_sha256"]), 64)


class JobJournalTests(unittest.TestCase):
    def test_unfinished_job_is_marked_interrupted_after_restart(self):
        with TemporaryDirectory() as directory:
            first = JobManager(directory)
            job = first.create("sample", {"n_qubits": 2}, seed=3)
            second = JobManager(directory)
            restored = second.get(job.job_id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.status, "failed")
            self.assertIn("restarted", restored.error)


if __name__ == "__main__":
    unittest.main()
