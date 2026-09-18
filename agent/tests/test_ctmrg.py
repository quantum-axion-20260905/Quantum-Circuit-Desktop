import math
import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from qc_agent.core.ctmrg import run_ctmrg
from qc_agent.plugins.models import CTMRGPayload, IPEPSInteraction, PauliTerm


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


if __name__ == "__main__":
    unittest.main()
