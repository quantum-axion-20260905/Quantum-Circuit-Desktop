import math
import unittest

import numpy as np

from qc_agent.core.finite_peps_optimizer import finite_torus_energy_gradient, run_finite_torus_gradient
from qc_agent.plugins.models import CTMRGPayload, IPEPSInteraction, PauliTerm


def _payload() -> CTMRGPayload:
    return CTMRGPayload(
        unit_cell=[1, 1],
        virtual_bond_dim=2,
        dtype="complex128",
        full_update_max_parameters=128,
        full_update_max_evaluations=64,
        full_update_step=0.1,
        interactions=[IPEPSInteraction(
            left_site=0,
            right_site=0,
            displacement=[1, 0],
            left_pauli="Z",
            right_pauli="Z",
            coefficient=0.0,
        )],
        terms=[PauliTerm(paulis={0: "Z"}, coefficient=-1.0)],
    )


class FinitePEPSOptimizerTests(unittest.TestCase):
    def test_analytic_gradient_matches_real_and_imaginary_finite_difference(self):
        payload = _payload()
        rng = np.random.default_rng(3)
        tensor = (rng.normal(size=(2, 2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2, 2))) * 0.2
        result = finite_torus_energy_gradient(np, payload, [tensor], with_gradient=True)
        flat = tensor.reshape(-1)
        index = 7
        epsilon = 1e-6
        gradient = result["gradients"][0].reshape(-1)[index]
        for imaginary, analytic in ((False, gradient.real), (True, gradient.imag)):
            plus = flat.copy()
            minus = flat.copy()
            delta = (1j if imaginary else 1.0) * epsilon
            plus[index] += delta
            minus[index] -= delta
            plus_energy = finite_torus_energy_gradient(np, payload, [plus.reshape(tensor.shape)], with_gradient=False)["energy"]
            minus_energy = finite_torus_energy_gradient(np, payload, [minus.reshape(tensor.shape)], with_gradient=False)["energy"]
            numeric = (plus_energy - minus_energy) / (2.0 * epsilon)
            self.assertAlmostEqual(numeric, analytic, places=5)

    def test_finite_torus_gradient_reduces_exact_reference_energy(self):
        payload = _payload().model_copy(update={"optimization_steps": 6})
        tensor = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        tensor[:, 0, 0, 0, 0] = 1.0 / math.sqrt(2.0)
        result = run_finite_torus_gradient(np, payload, [tensor])
        self.assertEqual(result["optimizer"], "finite-torus-gradient")
        self.assertEqual(result["gradient_backend"], "analytic-finite-torus-reverse-contraction")
        self.assertTrue(result["materializes_reference_statevector"])
        self.assertLess(result["final_energy"], result["initial_energy"] - 3.0)
        self.assertLess(result["final_variance"], 0.01)
        self.assertFalse(result["evaluation_budget_exhausted"])


if __name__ == "__main__":
    unittest.main()
