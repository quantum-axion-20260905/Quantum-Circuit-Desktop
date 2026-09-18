import math
import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from qc_agent.core.ctmrg import run_ctmrg, run_ctmrg_convergence_study
from qc_agent.core.peps import PEPSRuntime
from qc_agent.plugins.lattice import build_ctmrg_spin_payload
from qc_agent.plugins.models import CTMRGConvergenceStudyPayload, CTMRGPayload, IPEPSInteraction, LatticeHamiltonianPayload, LatticeSpec, PEPSPayload, PauliTerm


class CTMRGTests(unittest.TestCase):
    def test_product_up_state_contracts_z_and_interaction_without_statevector(self):
        payload = CTMRGPayload(
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0, label="magnetization")],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=0.5,
                label="zz",
            )],
            environment_bond_dim=2,
            iterations=4,
            tolerance=1e-10,
        )
        result = run_ctmrg(np, payload)
        self.assertEqual(result["backend"], "tensor-network-ctmrg")
        self.assertAlmostEqual(result["observables"][0]["value"], 1.0, places=6)
        self.assertAlmostEqual(result["interactions"][0]["value"], 1.0, places=6)
        self.assertAlmostEqual(result["energy"], 1.5, places=6)
        self.assertTrue(math.isfinite(result["residual"]))
        self.assertTrue(math.isfinite(result["correlation_length"]))
        self.assertEqual(len(result["environment_spectrum"]), 1)
        self.assertGreaterEqual(len(result["environment_spectrum"][0]), 1)
        self.assertTrue(result["reference_validation"]["performed"])
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertAlmostEqual(result["reference_validation"]["energy_error"], 0.0, places=8)
        self.assertAlmostEqual(result["energy_variance"], 0.0, places=8)
        self.assertFalse(result["resource_estimate"]["materializes_statevector"])
        self.assertEqual(result["research_result"]["status"], "needs_review")

    def test_plus_state_has_unit_x_expectation(self):
        payload = CTMRGPayload(
            terms=[PauliTerm(paulis={0: "X"}, coefficient=1.0)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[0, 1],
                left_pauli="X",
                right_pauli="X",
                coefficient=1.0,
            )],
            initial_state="plus",
            environment_bond_dim=1,
            iterations=2,
        )
        result = run_ctmrg(np, payload)
        self.assertAlmostEqual(result["observables"][0]["value"], 1.0, places=6)
        self.assertAlmostEqual(result["interactions"][0]["value"], 1.0, places=6)

    def test_entangled_ghz_tensor_keeps_two_site_order_parameter(self):
        """A D=2 symmetry-degenerate tensor must not collapse to a 0/0 bond."""

        tensor_data: list[list[float]] = []
        for physical in range(2):
            for up in range(2):
                for down in range(2):
                    for left in range(2):
                        for right in range(2):
                            tensor_data.append([
                                float(physical == up == down == left == right),
                                0.0,
                            ])
        result = run_ctmrg(np, CTMRGPayload(
            unit_cell=[1, 1],
            virtual_bond_dim=2,
            dtype="complex64",
            tensor_data=tensor_data,
            environment_bond_dim=2,
            iterations=8,
            tolerance=1e-6,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
        ))
        # The symmetric GHZ transfer fixed point has <Z>=0 and <Z_i Z_j>=1.
        self.assertAlmostEqual(result["observables"][0]["value"], 0.0, places=5)
        self.assertAlmostEqual(result["interactions"][0]["value"], 1.0, places=5)
        self.assertAlmostEqual(result["energy"], 1.0, places=5)
        self.assertIsNone(result["correlation_length"])
        self.assertTrue(result["reference_validation"]["performed"])
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertEqual(result["reference_validation"]["reference"], "analytic-ghz-transfer-fixed-point")

    def test_two_site_checkerboard_contracts_neel_bond(self):
        payload = CTMRGPayload(
            unit_cell=[2, 1],
            initial_state="neel",
            terms=[
                PauliTerm(paulis={0: "Z"}, coefficient=1.0),
                PauliTerm(paulis={1: "Z"}, coefficient=1.0),
            ],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=1,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
            environment_bond_dim=2,
            iterations=3,
        )
        result = run_ctmrg(np, payload)
        self.assertEqual(result["unit_cell"], [2, 1])
        self.assertEqual(result["unit_cell_sites"], 2)
        self.assertAlmostEqual(result["observables"][0]["value"], 1.0, places=6)
        self.assertAlmostEqual(result["observables"][1]["value"], -1.0, places=6)
        self.assertAlmostEqual(result["interactions"][0]["value"], -1.0, places=6)
        self.assertTrue(result["energy_complete"])

    def test_vertical_checkerboard_and_multi_tensor_import(self):
        # Two D=1 tensors are serialized site-major: |up> followed by |down>.
        tensor_data = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
        payload = CTMRGPayload(
            unit_cell=[1, 2],
            tensor_data=tensor_data,
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=1,
                displacement=[0, 1],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
            environment_bond_dim=2,
            iterations=2,
        )
        result = run_ctmrg(np, payload)
        self.assertEqual(result["unit_cell"], [1, 2])
        self.assertEqual(result["tensor_source"], "imported")
        self.assertAlmostEqual(result["interactions"][0]["value"], -1.0, places=6)

    def test_two_by_two_periodic_cell_contracts_all_nearest_bonds(self):
        interactions = [
            IPEPSInteraction(left_site=0, right_site=1, displacement=[1, 0], left_pauli="Z", right_pauli="Z", coefficient=1.0),
            IPEPSInteraction(left_site=0, right_site=2, displacement=[0, 1], left_pauli="Z", right_pauli="Z", coefficient=1.0),
            IPEPSInteraction(left_site=1, right_site=3, displacement=[0, 1], left_pauli="Z", right_pauli="Z", coefficient=1.0),
            IPEPSInteraction(left_site=2, right_site=3, displacement=[1, 0], left_pauli="Z", right_pauli="Z", coefficient=1.0),
        ]
        result = run_ctmrg(np, CTMRGPayload(
            unit_cell=[2, 2],
            initial_state="neel",
            interactions=interactions,
            environment_bond_dim=2,
            iterations=2,
        ))
        self.assertEqual(result["unit_cell_sites"], 4)
        self.assertEqual([round(item["value"], 6) for item in result["interactions"]], [-1.0] * 4)
        self.assertTrue(result["energy_complete"])
        self.assertFalse(result["resource_estimate"]["materializes_statevector"])
        self.assertEqual(len(result["correlation_lengths_by_site"]), 4)
        self.assertEqual(len(result["environment_spectrum"]), 4)

    def test_two_by_two_product_ctmrg_matches_independent_finite_peps_reference(self):
        finite_payload = PEPSPayload(
            n_qubits=4,
            terms=[PauliTerm(paulis={0: "Z", 1: "Z"}, coefficient=1.0)],
            lattice=LatticeSpec(dimensions=[2, 2]),
            contraction_method="opt_einsum",
        )
        finite_runtime = PEPSRuntime(np, finite_payload)
        flip = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        finite_runtime.apply_one_site(1, flip)
        finite_runtime.apply_one_site(2, flip)
        finite_reference = finite_runtime._contract_double_layer({0: "Z", 1: "Z"})

        ctmrg = run_ctmrg(np, CTMRGPayload(
            unit_cell=[2, 2],
            initial_state="neel",
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=1,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
            environment_bond_dim=2,
            iterations=2,
        ))
        self.assertAlmostEqual(finite_reference, -1.0, places=6)
        self.assertAlmostEqual(ctmrg["interactions"][0]["value"], finite_reference, places=6)
        self.assertTrue(ctmrg["reference_validation"]["passed"])

    def test_periodic_ising_builder_feeds_ctmrg_without_finite_geometry_leak(self):
        payload = build_ctmrg_spin_payload(
            LatticeHamiltonianPayload(dimensions=[2, 2], model="ising", coupling=1.0),
            initial_state="up",
            environment_bond_dim=2,
            iterations=2,
        )
        self.assertEqual(payload.unit_cell, [2, 2])
        self.assertEqual(len(payload.interactions), 8)
        self.assertTrue(all(item.displacement in ([1, 0], [0, 1]) for item in payload.interactions))
        result = run_ctmrg(np, payload)
        self.assertAlmostEqual(result["energy"], -8.0, places=6)
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertAlmostEqual(result["energy_variance"], 0.0, places=6)

    def test_periodic_heisenberg_builder_reports_product_limit_reference(self):
        payload = build_ctmrg_spin_payload(
            LatticeHamiltonianPayload(dimensions=[1, 1], model="heisenberg", coupling=1.0),
            initial_state="up",
            environment_bond_dim=1,
            iterations=2,
        )
        self.assertEqual(len(payload.interactions), 6)
        result = run_ctmrg(np, payload)
        self.assertAlmostEqual(result["energy"], 2.0, places=6)
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertAlmostEqual(result["energy_variance"], 0.0, places=6)

    def test_periodic_heisenberg_two_by_two_reference_covers_all_bonds(self):
        payload = build_ctmrg_spin_payload(
            LatticeHamiltonianPayload(dimensions=[2, 2], model="heisenberg", coupling=1.0),
            initial_state="up",
            environment_bond_dim=2,
            iterations=2,
        )
        result = run_ctmrg(np, payload)
        # The product-up state contributes only ZZ on all eight directed
        # periodic nearest-neighbor bonds in the explicit unit-cell contract.
        self.assertEqual(len(payload.interactions), 24)
        self.assertAlmostEqual(result["energy"], 8.0, places=6)
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertAlmostEqual(result["reference_validation"]["interaction_max_abs_error"], 0.0, places=8)

    def test_two_by_two_checkpoint_resume_restores_all_environments(self):
        payload = CTMRGPayload(
            unit_cell=[2, 2],
            initial_state="neel",
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=1,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
            environment_bond_dim=2,
            iterations=2,
            tolerance=1e-20,
        )
        with TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, "ctmrg-2x2.npz")
            partial = run_ctmrg(np, payload.model_copy(update={"checkpoint_path": checkpoint_path}))
            self.assertEqual(partial["checkpoint"]["metadata"]["environment_count"], 4)
            resumed = run_ctmrg(np, payload.model_copy(update={
                "iterations": 4,
                "resume_from": checkpoint_path,
                "checkpoint_path": checkpoint_path,
            }))
            fresh = run_ctmrg(np, payload.model_copy(update={"iterations": 4}))
            self.assertEqual(resumed["unit_cell"], [2, 2])
            self.assertEqual(resumed["checkpoint"]["metadata"]["environment_count"], 4)
            self.assertAlmostEqual(resumed["energy"], fresh["energy"], places=6)

    def test_product_coordinate_descent_optimizes_mean_field_energy(self):
        payload = CTMRGPayload(
            initial_state="plus",
            optimization="product-coordinate-descent",
            optimization_steps=16,
            optimization_tolerance=1e-8,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=-0.2)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=-1.0,
            )],
            environment_bond_dim=1,
            iterations=1,
        )
        result = run_ctmrg(np, payload)
        diagnostics = result["optimization_diagnostics"]
        self.assertEqual(result["method"], "ipeps-ctmrg-product-optimization")
        self.assertLess(diagnostics["final_energy"], diagnostics["initial_energy"] - 0.5)
        self.assertAlmostEqual(result["energy"], -1.2, places=6)
        self.assertTrue(any("mean-field baseline" in warning for warning in result["warnings"]))

    def test_product_optimizer_rejects_entangled_virtual_bond(self):
        payload = CTMRGPayload(
            virtual_bond_dim=2,
            optimization="product-coordinate-descent",
            interactions=[{
                "left_site": 0,
                "right_site": 0,
                "displacement": [1, 0],
                "left_pauli": "Z",
                "right_pauli": "Z",
                "coefficient": 1.0,
            }],
        )
        with self.assertRaisesRegex(ValueError, "virtual_bond_dim=1"):
            run_ctmrg(np, payload)

    def test_simple_update_runs_entangled_tensor_baseline_without_statevector(self):
        result = run_ctmrg(np, CTMRGPayload(
            unit_cell=[2, 1],
            virtual_bond_dim=2,
            initial_state="plus",
            optimization="simple-update",
            optimization_steps=3,
            optimization_dt=0.05,
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=1,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=-1.0,
            )],
            environment_bond_dim=2,
            iterations=2,
        ))
        self.assertEqual(result["method"], "ipeps-simple-update-ctmrg")
        self.assertEqual(result["optimization"], "simple-update")
        self.assertEqual(result["optimization_diagnostics"]["bond_dim_history"][-1], 2)
        self.assertTrue(math.isfinite(result["energy"]))
        self.assertFalse(result["resource_estimate"]["materializes_statevector"])
        self.assertTrue(any("full-update" in warning for warning in result["warnings"]))

    def test_full_update_recomputes_ctmrg_energy_for_bounded_tensor_trials(self):
        result = run_ctmrg(np, CTMRGPayload(
            initial_state="plus",
            optimization="full-update",
            optimization_steps=1,
            full_update_step=0.1,
            full_update_max_parameters=8,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=-0.2)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=-1.0,
            )],
            environment_bond_dim=1,
            iterations=1,
        ))
        diagnostics = result["optimization_diagnostics"]
        self.assertEqual(result["method"], "ipeps-full-update-ctmrg")
        self.assertEqual(diagnostics["parameter_count"], 4)
        self.assertGreater(diagnostics["evaluations"], 1)
        self.assertLess(diagnostics["final_energy"], diagnostics["initial_energy"])
        self.assertTrue(any("CTMRG energy" in warning for warning in result["warnings"]))

    def test_finite_difference_gradient_full_update_is_explicitly_bounded(self):
        result = run_ctmrg(np, CTMRGPayload(
            initial_state="plus",
            optimization="full-update",
            full_update_optimizer="finite-difference-gradient",
            optimization_steps=1,
            full_update_step=0.1,
            full_update_gradient_epsilon=1e-3,
            full_update_max_parameters=8,
            full_update_max_evaluations=128,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=-0.2)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=-1.0,
            )],
            environment_bond_dim=1,
            iterations=1,
        ))
        diagnostics = result["optimization_diagnostics"]
        self.assertEqual(result["method"], "ipeps-full-update-gradient-ctmrg")
        self.assertEqual(diagnostics["optimizer"], "finite-difference-gradient")
        self.assertGreater(diagnostics["evaluations"], 1)
        self.assertLess(diagnostics["final_energy"], diagnostics["initial_energy"])
        self.assertTrue(any("finite-difference-gradient" in warning for warning in result["warnings"]))

    def test_spsa_full_update_uses_parameter_independent_objective_budget(self):
        result = run_ctmrg(np, CTMRGPayload(
            initial_state="plus",
            optimization="full-update",
            full_update_optimizer="spsa-gradient",
            optimization_steps=2,
            full_update_step=0.1,
            full_update_gradient_epsilon=1e-3,
            full_update_max_parameters=8,
            full_update_max_evaluations=32,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=-0.2)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=-1.0,
            )],
            environment_bond_dim=1,
            iterations=1,
        ))
        diagnostics = result["optimization_diagnostics"]
        self.assertEqual(result["method"], "ipeps-full-update-spsa-ctmrg")
        self.assertEqual(diagnostics["optimizer"], "spsa-gradient")
        self.assertEqual(diagnostics["gradient_backend"], "deterministic-simultaneous-perturbation")
        self.assertLessEqual(diagnostics["evaluations"], 1 + 2 * 6)
        self.assertTrue(any("SPSA" in warning for warning in result["warnings"]))

    def test_full_update_parameter_admission_is_explicit(self):
        payload = CTMRGPayload(
            virtual_bond_dim=2,
            optimization="full-update",
            full_update_max_parameters=4,
            interactions=[{
                "left_site": 0,
                "right_site": 0,
                "displacement": [1, 0],
                "left_pauli": "Z",
                "right_pauli": "Z",
                "coefficient": 1.0,
            }],
        )
        with self.assertRaisesRegex(ValueError, "full-update tensor parameter count"):
            run_ctmrg(np, payload)

    def test_imported_complex_tensor_uses_declared_virtual_bond(self):
        tensor = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        tensor[0, 0, 0, 0, 0] = 1.0 / math.sqrt(2.0)
        tensor[1, 1, 1, 1, 1] = 1.0 / math.sqrt(2.0)
        payload = CTMRGPayload(
            virtual_bond_dim=2,
            tensor_data=[[float(value.real), float(value.imag)] for value in tensor.reshape(-1)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
            environment_bond_dim=2,
            iterations=3,
        )
        result = run_ctmrg(np, payload)
        self.assertEqual(result["tensor_source"], "imported")
        self.assertEqual(result["virtual_bond_dim"], 2)
        self.assertFalse(result["resource_estimate"]["materializes_statevector"])
        self.assertAlmostEqual(result["interactions"][0]["value"], 1.0, places=5)
        self.assertTrue(result["reference_validation"]["performed"])
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertEqual(result["reference_validation"]["reference"], "analytic-ghz-transfer-fixed-point")
        self.assertIsNone(result["energy_variance"])
        self.assertFalse(any("withheld" in warning for warning in result["warnings"]))

    def test_checkpoint_resume_restores_environment_and_history(self):
        payload = CTMRGPayload(
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
            environment_bond_dim=2,
            iterations=2,
            tolerance=1e-20,
        )
        with TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, "ctmrg.npz")
            partial = run_ctmrg(np, payload.model_copy(update={"checkpoint_path": checkpoint_path}))
            self.assertTrue(os.path.exists(checkpoint_path))
            self.assertEqual(partial["checkpoint"]["schema"], "quantum-circuit/checkpoint-v1")
            resumed = run_ctmrg(np, payload.model_copy(update={
                "iterations": 4,
                "resume_from": checkpoint_path,
                "checkpoint_path": checkpoint_path,
            }))
            fresh = run_ctmrg(np, payload.model_copy(update={"iterations": 4}))
            self.assertGreaterEqual(resumed["iterations"], partial["iterations"])
            self.assertAlmostEqual(resumed["energy"], fresh["energy"], places=6)
            self.assertEqual(resumed["research_result"]["checkpoint"]["request_sha256"], partial["checkpoint"]["request_sha256"])

    def test_tensor_data_shape_and_finite_validation_is_explicit(self):
        with self.assertRaises(ValueError):
            CTMRGPayload(
                virtual_bond_dim=2,
                tensor_data=[[0.0, 0.0]],
                interactions=[{
                    "left_site": 0,
                    "right_site": 0,
                    "displacement": [1, 0],
                    "left_pauli": "Z",
                    "right_pauli": "Z",
                    "coefficient": 1.0,
                }],
            )
        with self.assertRaises(ValueError):
            CTMRGPayload(
                tensor_data=[[float("nan"), 0.0] for _ in range(2)],
                interactions=[{
                    "left_site": 0,
                    "right_site": 0,
                    "displacement": [1, 0],
                    "left_pauli": "Z",
                    "right_pauli": "Z",
                    "coefficient": 1.0,
                }],
            )

    def test_environment_dimension_convergence_study_is_bounded_and_replayable(self):
        payload = CTMRGPayload(
            initial_state="plus",
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="X",
                right_pauli="X",
                coefficient=-1.0,
            )],
            environment_bond_dim=4,
            iterations=2,
        )
        study = run_ctmrg_convergence_study(np, payload, [1, 2, 4])
        self.assertEqual(study["method"], "ipeps-ctmrg-environment-convergence-study")
        self.assertEqual([point["environment_bond_dim"] for point in study["points"]], [1, 2, 4])
        self.assertFalse(study["materializes_statevector"])
        self.assertIsNone(study["points"][0]["energy_delta"])
        for point in study["points"]:
            self.assertTrue(math.isfinite(point["energy"]))
            self.assertTrue(math.isfinite(point["residual"]))
            self.assertTrue(math.isfinite(point["correlation_length"]))
            self.assertGreaterEqual(len(point["environment_spectrum"]), 1)

    def test_environment_dimension_convergence_study_requires_plain_contraction(self):
        with self.assertRaisesRegex(ValueError, "optimization='none'"):
            run_ctmrg_convergence_study(np, CTMRGPayload(
                optimization="simple-update",
                interactions=[{
                    "left_site": 0,
                    "right_site": 0,
                    "displacement": [1, 0],
                    "left_pauli": "Z",
                    "right_pauli": "Z",
                    "coefficient": 1.0,
                }],
            ), [1, 2])

    def test_environment_dimension_study_reports_progress_and_honors_cancellation(self):
        payload = CTMRGPayload(
            interactions=[{
                "left_site": 0,
                "right_site": 0,
                "displacement": [1, 0],
                "left_pauli": "Z",
                "right_pauli": "Z",
                "coefficient": 1.0,
            }],
            environment_bond_dim=2,
            iterations=2,
        )
        progress = []
        run_ctmrg_convergence_study(
            np,
            payload,
            [1, 2],
            progress_cb=lambda value, phase: progress.append((value, phase)),
        )
        self.assertTrue(progress)
        self.assertLessEqual(max(value for value, _ in progress), 1.0)
        self.assertTrue(any("chi-1" in phase for _, phase in progress))
        with self.assertRaisesRegex(RuntimeError, "job canceled"):
            run_ctmrg_convergence_study(np, payload, [1, 2], cancel_cb=lambda: True)

    def test_environment_dimension_study_api_contract_rejects_optimized_problem(self):
        with self.assertRaisesRegex(ValueError, "problem.optimization='none'"):
            CTMRGConvergenceStudyPayload(
                problem=CTMRGPayload(
                    optimization="simple-update",
                    interactions=[{
                        "left_site": 0,
                        "right_site": 0,
                        "displacement": [1, 0],
                        "left_pauli": "Z",
                        "right_pauli": "Z",
                        "coefficient": 1.0,
                    }],
                ),
                environment_bond_dims=[1, 2],
            )


if __name__ == "__main__":
    unittest.main()
