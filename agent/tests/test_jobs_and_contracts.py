import threading
import time
import unittest
from tempfile import TemporaryDirectory

from qc_agent.jobs import JobManager, ResourceRequest
from qc_agent.plugins.base import PluginInfo, plugin_actions, validate_plugin


class JobResourceTests(unittest.TestCase):
    def test_gpu_reservation_serializes_exclusive_jobs(self):
        with TemporaryDirectory() as directory:
            manager = JobManager(
                directory,
                max_workers=2,
                resource_provider=lambda: {
                    "gpu": {
                        "available": True,
                        "count": 1,
                        "device0": {"free_global_mem": 1024 * 1024 * 1024},
                    }
                },
            )
            request = ResourceRequest(kind="cuda", memory_mb=128, exclusive=True)
            first = manager.create("first", {}, None, resource=request)
            second = manager.create("second", {}, None, resource=request)
            active = 0
            maximum = 0
            lock = threading.Lock()

            def work(job):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
                return {"metrics": {"ok": True}}

            manager.run_async(first, work, resource=request)
            manager.run_async(second, work, resource=request)
            deadline = time.time() + 3
            while time.time() < deadline and (first.status not in ("done", "failed") or second.status not in ("done", "failed")):
                time.sleep(0.01)
            self.assertEqual(first.status, "done")
            self.assertEqual(second.status, "done")
            self.assertEqual(maximum, 1)
            self.assertEqual(first.resource["assigned_device"], 0)
            self.assertEqual(second.resource["assigned_device"], 0)
            deadline = time.time() + 1
            while time.time() < deadline and manager.broker.snapshot()["reserved_gpu_memory_mb"]:
                time.sleep(0.01)
            self.assertEqual(manager.broker.snapshot()["reserved_gpu_memory_mb"], {})
            manager.shutdown(wait=True)

    def test_queued_gpu_job_can_be_canceled_without_running(self):
        with TemporaryDirectory() as directory:
            manager = JobManager(
                directory,
                max_workers=1,
                resource_provider=lambda: {"gpu": {"available": False}},
            )
            request = ResourceRequest(kind="cuda", memory_mb=1, exclusive=True)
            job = manager.create("blocked", {}, None, resource=request)
            manager.run_async(job, lambda active: {"metrics": {"ran": True}}, resource=request, acquire_timeout_s=2)
            time.sleep(0.05)
            self.assertTrue(manager.cancel(job.job_id))
            time.sleep(0.1)
            self.assertEqual(job.status, "canceled")
            manager.shutdown(wait=True)


class PluginContractTests(unittest.TestCase):
    def test_plugin_contract_exposes_actions_and_api_version(self):
        class Example:
            info = PluginInfo("example", "Example", "1.0.0", "test", ("lattice-preview",))

            def preview_lattice(self, payload):
                return payload

        plugin = Example()
        validate_plugin(plugin)
        self.assertEqual(plugin_actions(plugin), ("preview_lattice",))

    def test_invalid_plugin_is_rejected(self):
        class Invalid:
            info = PluginInfo("bad", "Bad", "1.0.0", "test", ())

        with self.assertRaises(ValueError):
            validate_plugin(Invalid())


class UnifiedApiTests(unittest.TestCase):
    def test_dynamic_ctmrg_endpoint_is_explicitly_registered(self):
        from qc_agent.server import app

        paths = {route.path for route in app.routes if hasattr(route, "path")}
        self.assertIn("/jobs/ctmrg/dynamic", paths)
        self.assertIn("/jobs/ctmrg/dynamic/convergence", paths)
        self.assertIn("/jobs/ctmrg/dynamic/sectors", paths)

    def test_ctmrg_study_preflight_prices_sequential_points(self):
        from qc_agent.server import _scale_ctmrg_study_preflight

        ready = _scale_ctmrg_study_preflight(
            {"feasible": True, "status": "ready", "estimated_time_ms": 11, "warnings": []},
            point_count=4,
            max_time_ms=50,
        )
        self.assertEqual(ready["estimated_point_time_ms"], 11)
        self.assertEqual(ready["estimated_total_time_ms"], 44)
        self.assertEqual(ready["estimated_time_ms"], 44)
        self.assertTrue(ready["feasible"])

        rejected = _scale_ctmrg_study_preflight(
            {"feasible": True, "status": "ready", "estimated_time_ms": 11, "warnings": []},
            point_count=5,
            max_time_ms=50,
        )
        self.assertFalse(rejected["feasible"])
        self.assertEqual(rejected["status"], "rejected")
        self.assertTrue(any("study time" in warning for warning in rejected["warnings"]))

    def test_async_physics_defaults_resolve_tensor_network_backend(self):
        from qc_agent.server import _async_backend, _async_parse

        terms = [{"paulis": {"0": "Z"}, "coefficient": 1.0}]
        tebd = _async_parse("tebd", {
            "n_qubits": 2,
            "terms": terms,
            "dt": 0.01,
            "steps": 1,
            "bond_dim": 2,
        })
        peps = _async_parse("peps", {
            "n_qubits": 4,
            "terms": terms,
            "lattice": {"dimensions": [2, 2], "boundary": "open"},
            "dt": 0.01,
            "steps": 1,
            "bond_dim": 2,
        })
        self.assertEqual(_async_backend("tebd", tebd), ("tensor-network", "evolve"))
        self.assertEqual(_async_backend("peps", peps), ("tensor-network", "peps"))

    def test_boundary_mps_ctmrg_study_has_unified_async_contract(self):
        from qc_agent.server import _async_backend, _async_parse

        payload = _async_parse("ctmrg_boundary_mps_convergence", {
            "problem": {
                "environment_bond_dim": 2,
                "iterations": 2,
                "interactions": [{
                    "left_site": 0,
                    "right_site": 0,
                    "displacement": [1, 0],
                    "left_pauli": "Z",
                    "right_pauli": "Z",
                    "coefficient": 1.0,
                }],
            },
            "patch_sizes": [[2, 2], [3, 3]],
            "boundary_bond_dims": [2, 4],
        })
        self.assertEqual(payload.__class__.__name__, "CTMRGBoundaryMPSStudyPayload")
        self.assertEqual(_async_backend("ctmrg_boundary_mps_convergence", payload), ("tensor-network", "ctmrg"))

    def test_gpu_benchmark_preflight_accepts_generic_budget(self):
        from qc_agent.server import _async_backend, _async_parse, _async_preflight

        payload = _async_parse("bench_matmul", {
            "size": 256,
            "iters": 1,
            "dtype": "fp16",
        })
        resolved, operation = _async_backend("bench_matmul", payload)
        report = _async_preflight(
            "bench_matmul",
            payload,
            resolved,
            {"max_qubits": 1, "max_mem_mb": 128, "max_time_ms": 30000},
        )
        self.assertEqual((resolved, operation), ("statevector", "simulate"))
        self.assertTrue(report["feasible"])
        self.assertEqual(report["method"], "gpu-matmul-bound")

    def test_reference_run_uses_unified_job_lifecycle(self):
        from qc_agent.server import AsyncSubmission, get_job, submit_unified_async_route

        submitted = submit_unified_async_route(AsyncSubmission(
            kind="run",
            payload={
                "n_qubits": 1,
                "gates": [],
                "backend": "reference",
                "result_type": "samples",
                "shots": 4,
            },
        ))
        deadline = time.time() + 3
        current = get_job(submitted["job_id"])
        while current["status"] in ("queued", "running") and time.time() < deadline:
            time.sleep(0.01)
            current = get_job(submitted["job_id"])
        self.assertEqual(current["status"], "done")
        self.assertEqual(sum(current["artifacts"]["result"]["counts"].values()), 4)
        self.assertIn("problem_sha256", current["artifacts"]["result"]["provenance"])
        self.assertIn("preflight", current["artifacts"]["result"])

    def test_structured_observable_contract_is_named_and_sparse(self):
        from types import SimpleNamespace
        from qc_agent.core.observables import structured_observables

        result = structured_observables(
            [
                SimpleNamespace(paulis={1: "Z", 0: "X"}, coefficient=-0.5, label="bond"),
                SimpleNamespace(paulis={}, coefficient=1.0, label=None),
            ],
            [0.25, 1.0],
        )
        self.assertEqual(result[0]["label"], "bond")
        self.assertEqual(result[0]["paulis"], {"0": "X", "1": "Z"})
        self.assertEqual(result[0]["coefficient"], -0.5)
        self.assertEqual(result[1]["label"], "I")


if __name__ == "__main__":
    unittest.main()
