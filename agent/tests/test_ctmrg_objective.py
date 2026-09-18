import unittest

import numpy as np

from qc_agent.core.ctmrg import run_ctmrg
from qc_agent.core.ctmrg_objective import CTMRGObjective
from qc_agent.plugins.models import IPEPSInteraction, CTMRGPayload


class CTMRGObjectiveTests(unittest.TestCase):
    def test_objective_uses_plain_ctmrg_and_enforces_shared_budget(self):
        payload = CTMRGPayload(
            unit_cell=[1, 1],
            virtual_bond_dim=1,
            dtype="complex128",
            initial_state="plus",
            environment_bond_dim=1,
            iterations=1,
            full_update_max_evaluations=8,
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="X",
                right_pauli="X",
                coefficient=-1.0,
            )],
        )
        objective = CTMRGObjective(np, payload, run_ctmrg)
        tensor = np.ones((2, 1, 1, 1, 1), dtype=np.complex128)

        result = objective.evaluate([tensor])

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["energy"], -1.0, places=6)
        self.assertEqual(objective.evaluations, 1)
        self.assertEqual(objective.remaining_evaluations, 7)
        self.assertEqual(objective.payload.optimization, "none")
        self.assertIsNone(objective.payload.checkpoint_path)
        self.assertIsNone(objective.payload.optimizer_checkpoint_path)
        for _ in range(7):
            self.assertIsNotNone(objective.evaluate([tensor]))
        self.assertEqual(objective.remaining_evaluations, 0)
        self.assertIsNone(objective.evaluate([tensor]))


if __name__ == "__main__":
    unittest.main()
