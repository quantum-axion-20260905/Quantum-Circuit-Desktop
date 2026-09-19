import json
import unittest

from qc_agent.backends.preflight import estimate_ctmrg
from qc_agent.api.contracts import AsyncSubmission
from qc_agent.core.contracts import (
    CHECKPOINT_SCHEMA,
    CapabilityError,
    RESULT_SCHEMA,
    CheckpointManifest,
    ConvergencePoint,
    ConvergenceReport,
    ResearchResult,
    ResourceBudget,
    TruncationReport,
)
from qc_agent.plugins.models import CTMRGPayload


class ComputeContractTests(unittest.TestCase):
    def test_capability_error_is_distinct_from_numerical_failure(self):
        with self.assertRaises(CapabilityError):
            raise CapabilityError("CTMRG is not available for a finite 3D representation")

    def test_resource_budget_rejects_unbounded_values(self):
        with self.assertRaises(ValueError):
            ResourceBudget(max_gpu_mb=0)
        with self.assertRaises(ValueError):
            ResourceBudget(max_time_ms=-1)

    def test_result_envelope_is_versioned_and_json_serializable(self):
        result = ResearchResult(
            status="needs_review",
            method="finite-peps-boundary-mps",
            representation="finite-peps",
            metrics={"energy": -1.25},
            truncation=TruncationReport(discarded_weight=1e-6, max_bond_dim=8),
            convergence=ConvergenceReport(
                criterion="energy spread < 1e-5",
                points=[ConvergencePoint(iteration=1, energy=-1.25, bond_dim=8)],
            ),
            warnings=["increase environment dimension"],
        )
        payload = result.to_dict()
        self.assertEqual(payload["schema"], RESULT_SCHEMA)
        json.dumps(payload)
        self.assertEqual(payload["convergence"]["points"][0]["bond_dim"], 8)

    def test_checkpoint_manifest_requires_request_digest(self):
        with self.assertRaises(ValueError):
            CheckpointManifest(
                checkpoint_id="cp-1",
                request_sha256="bad",
                method="dmrg",
                representation="mps",
                dtype="complex64",
                device="cuda:0",
                step=1,
                created_at="2026-09-17T00:00:00Z",
            )
        manifest = CheckpointManifest(
            checkpoint_id="cp-1",
            request_sha256="a" * 64,
            method="dmrg",
            representation="mps",
            dtype="complex64",
            device="cuda:0",
            step=1,
            created_at="2026-09-17T00:00:00Z",
        )
        self.assertEqual(manifest.to_dict()["schema"], CHECKPOINT_SCHEMA)

    def test_ctmrg_contract_keeps_infinite_geometry_explicit(self):
        payload = CTMRGPayload(
            unit_cell=[1, 1],
            interactions=[
                {
                    "left_site": 0,
                    "right_site": 0,
                    "displacement": [1, 0],
                    "left_pauli": "Z",
                    "right_pauli": "Z",
                    "coefficient": -1.0,
                }
            ],
        )
        report = estimate_ctmrg(payload, gpu_free_mb=4096)
        self.assertTrue(report["feasible"])
        self.assertEqual(report["representation"], "ipeps")
        self.assertFalse(report["materializes_statevector"])
        supported = estimate_ctmrg(payload.model_copy(update={"unit_cell": [2, 1]}), gpu_free_mb=4096)
        self.assertTrue(supported["feasible"])
        two_by_two = estimate_ctmrg(payload.model_copy(update={"unit_cell": [2, 2]}), gpu_free_mb=4096)
        self.assertTrue(two_by_two["feasible"])
        unsupported = estimate_ctmrg(payload.model_copy(update={"unit_cell": [3, 1]}), gpu_free_mb=4096)
        self.assertFalse(unsupported["feasible"])
        with self.assertRaises(ValueError):
            CTMRGPayload(
                unit_cell=[2, 2],
                interactions=[
                    {
                        "left_site": 0,
                        "right_site": 4,
                        "displacement": [1, 0],
                        "left_pauli": "Z",
                        "right_pauli": "Z",
                        "coefficient": -1.0,
                    }
                ],
            )

    def test_ctmrg_preflight_prices_boundary_mps_transfer_fixed_point(self):
        payload = CTMRGPayload(
            boundary_mps_transfer_fixed_point=True,
            boundary_mps_width=2,
            boundary_mps_bond_dim=2,
            boundary_mps_transfer_cycles=3,
            interactions=[{
                "left_site": 0,
                "right_site": 0,
                "displacement": [1, 0],
                "left_pauli": "Z",
                "right_pauli": "Z",
                "coefficient": 1.0,
            }],
        )
        report = estimate_ctmrg(payload, gpu_free_mb=4096)
        self.assertTrue(report["feasible"])
        self.assertTrue(report["boundary_mps_transfer_fixed_point"])
        self.assertEqual(report["boundary_mps_transfer_cycles"], 3)
        self.assertTrue(any("transfer fixed-point" in warning for warning in report["warnings"]))

    def test_ctmrg_gradient_update_evaluation_budget_is_admitted_explicitly(self):
        payload = CTMRGPayload(
            optimization="full-update",
            full_update_optimizer="finite-difference-gradient",
            full_update_max_evaluations=8,
            interactions=[{
                "left_site": 0,
                "right_site": 0,
                "displacement": [1, 0],
                "left_pauli": "Z",
                "right_pauli": "Z",
                "coefficient": 1.0,
            }],
        )
        report = estimate_ctmrg(payload, gpu_free_mb=4096)
        self.assertFalse(report["feasible"])
        self.assertGreater(report["estimated_full_update_evaluations"], report["full_update_max_evaluations"])
        self.assertTrue(any("estimated evaluations" in warning for warning in report["blocking_warnings"]))

    def test_ctmrg_spsa_budget_scales_with_steps_not_parameters(self):
        payload = CTMRGPayload(
            optimization="full-update",
            full_update_optimizer="spsa-gradient",
            optimization_steps=3,
            full_update_max_evaluations=64,
            interactions=[{
                "left_site": 0,
                "right_site": 0,
                "displacement": [1, 0],
                "left_pauli": "Z",
                "right_pauli": "Z",
                "coefficient": 1.0,
            }],
        )
        report = estimate_ctmrg(payload, gpu_free_mb=4096)
        self.assertTrue(report["feasible"])
        self.assertEqual(report["estimated_full_update_evaluations"], 37)
        self.assertEqual(report["full_update_optimizer"], "spsa-gradient")

    def test_ctmrg_finite_torus_gradient_budget_and_reference_mode_are_explicit(self):
        payload = CTMRGPayload(
            optimization="full-update",
            full_update_optimizer="finite-torus-gradient",
            optimization_steps=3,
            full_update_max_evaluations=32,
            interactions=[{
                "left_site": 0,
                "right_site": 0,
                "displacement": [1, 0],
                "left_pauli": "Z",
                "right_pauli": "Z",
                "coefficient": 1.0,
            }],
        )
        report = estimate_ctmrg(payload, gpu_free_mb=4096)
        self.assertTrue(report["feasible"])
        self.assertEqual(report["estimated_full_update_evaluations"], 16)
        self.assertEqual(report["full_update_optimizer"], "finite-torus-gradient")
        self.assertTrue(report["optimization_materializes_reference_statevector"])
        self.assertTrue(any("finite reference statevector" in warning for warning in report["warnings"]))

    def test_ctmrg_convergence_is_a_first_class_async_kind(self):
        submission = AsyncSubmission(
            kind="ctmrg_convergence",
            payload={"problem": {"interactions": [{
                "left_site": 0,
                "right_site": 0,
                "displacement": [1, 0],
                "left_pauli": "Z",
                "right_pauli": "Z",
                "coefficient": 1.0,
            }]}, "environment_bond_dims": [1, 2]},
        )
        self.assertEqual(submission.kind, "ctmrg_convergence")

    def test_boundary_mps_transfer_convergence_is_a_first_class_async_kind(self):
        submission = AsyncSubmission(
            kind="ctmrg_boundary_mps_transfer_convergence",
            payload={
                "problem": {"interactions": [{
                    "left_site": 0,
                    "right_site": 0,
                    "displacement": [1, 0],
                    "left_pauli": "Z",
                    "right_pauli": "Z",
                    "coefficient": 1.0,
                }]},
                "widths": [1, 2],
                "boundary_bond_dims": [1, 2],
            },
        )
        self.assertEqual(submission.kind, "ctmrg_boundary_mps_transfer_convergence")

    def test_boundary_mps_transfer_gauge_covariance_is_a_first_class_async_kind(self):
        submission = AsyncSubmission(
            kind="ctmrg_boundary_mps_transfer_gauge_covariance",
            payload={
                "problem": {"virtual_bond_dim": 2},
                "widths": [1, 2],
                "boundary_bond_dims": [1, 4],
                "cycles": 4,
            },
        )
        self.assertEqual(submission.kind, "ctmrg_boundary_mps_transfer_gauge_covariance")


if __name__ == "__main__":
    unittest.main()
