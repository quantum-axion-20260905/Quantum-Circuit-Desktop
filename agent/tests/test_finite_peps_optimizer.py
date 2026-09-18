import math
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from qc_agent.core.finite_peps_optimizer import finite_torus_energy_gradient, run_finite_torus_gradient
from qc_agent.plugins.models import CTMRGPayload, IPEPSInteraction, PauliTerm


def _payload() -> CTMRGPayload:
    return CTMRGPayload(
        unit_cell=[1, 1],
        virtual_bond_dim=2,
        dtype="complex128",
        initial_state="plus",
        full_update_max_parameters=128,
        full_update_max_evaluations=64,
        full_update_step=0.1,
        environment_bond_dim=1,
        iterations=1,
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

    def test_finite_torus_gradient_checkpoint_resume_matches_fresh_run(self):
        from qc_agent.core.ctmrg import run_ctmrg

        with TemporaryDirectory() as directory:
            checkpoint = f"{directory}/finite-gradient.npz"
            base = _payload().model_dump(mode="json")
            partial_payload = CTMRGPayload(**{
                **base,
                "optimization": "full-update",
                "full_update_optimizer": "finite-torus-gradient",
                "optimization_steps": 2,
                "optimizer_checkpoint_path": checkpoint,
            })
            partial = run_ctmrg(np, partial_payload)
            self.assertTrue(partial["optimization_diagnostics"]["checkpoint"]["resumable"])
            self.assertEqual(partial["optimization_diagnostics"]["checkpoint"]["step"], 2)

            resumed_payload = CTMRGPayload(**{
                **base,
                "optimization": "full-update",
                "full_update_optimizer": "finite-torus-gradient",
                "optimization_steps": 6,
                "optimizer_checkpoint_path": checkpoint,
                "optimizer_resume_from": checkpoint,
            })
            resumed = run_ctmrg(np, resumed_payload)
            fresh_payload = CTMRGPayload(**{
                **base,
                "optimization": "full-update",
                "full_update_optimizer": "finite-torus-gradient",
                "optimization_steps": 6,
            })
            fresh = run_ctmrg(np, fresh_payload)
            resumed_diagnostics = resumed["optimization_diagnostics"]
            self.assertEqual(resumed_diagnostics["start_iteration"], 2)
            self.assertAlmostEqual(resumed_diagnostics["final_energy"], fresh["optimization_diagnostics"]["final_energy"], places=10)
            self.assertAlmostEqual(resumed["energy"], fresh["energy"], places=10)


if __name__ == "__main__":
    unittest.main()
