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

    def test_multi_site_cell_is_rejected_instead_of_silently_falling_back(self):
        payload = CTMRGPayload(
            unit_cell=[2, 1],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=1,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
        )
        with self.assertRaisesRegex(ValueError, "unit_cell=\[1, 1\]"):
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
