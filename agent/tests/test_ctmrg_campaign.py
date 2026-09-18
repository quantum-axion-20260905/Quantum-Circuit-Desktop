import unittest

from tools.ctmrg_campaign import (
    CELL_SHAPES,
    DEFAULT_ENVIRONMENT_DIMS,
    build_entangled_payload,
    build_product_payload,
)
from tools.ctmrg_gradient_gate import build_gradient_payload


class CTMRGCampaignTests(unittest.TestCase):
    def test_product_campaign_cases_cover_every_admitted_cell_shape(self):
        for cell in CELL_SHAPES:
            payload = build_product_payload(cell, iterations=2)
            self.assertEqual(payload.unit_cell, list(cell))
            self.assertEqual(payload.virtual_bond_dim, 1)
            self.assertEqual(len(payload.interactions), 2 * cell[0] * cell[1])
            self.assertEqual(payload.environment_bond_dim, max(DEFAULT_ENVIRONMENT_DIMS))
            for interaction in payload.interactions:
                left_x = interaction.left_site % cell[0]
                left_y = interaction.left_site // cell[0]
                dx, dy = interaction.displacement
                expected = ((left_x + dx) % cell[0]) + cell[0] * ((left_y + dy) % cell[1])
                self.assertEqual(interaction.right_site, expected)

    def test_entangled_campaign_case_has_explicit_2x2_tensor_contract(self):
        payload = build_entangled_payload(iterations=2)
        self.assertEqual(payload.unit_cell, [2, 2])
        self.assertEqual(payload.virtual_bond_dim, 2)
        self.assertEqual(len(payload.tensor_data or []), 4 * 2 * 2**4)
        self.assertEqual(len(payload.interactions), 2)
        self.assertEqual(payload.terms[0].paulis, {3: "Z"})

    def test_gradient_gate_case_has_explicit_entangled_contract(self):
        payload = build_gradient_payload("complex128", iterations=5, tolerance=1e-7)
        self.assertEqual(payload.unit_cell, [1, 1])
        self.assertEqual(payload.virtual_bond_dim, 2)
        self.assertEqual(payload.environment_bond_dim, 2)
        self.assertEqual(payload.full_update_optimizer, "autodiff-ctmrg-gradient")
        self.assertEqual(len(payload.interactions), 1)


if __name__ == "__main__":
    unittest.main()
