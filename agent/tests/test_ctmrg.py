import math
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


if __name__ == "__main__":
    unittest.main()
