import importlib.util
import math
import unittest

import numpy as np

from qc_agent.plugins.models import CTMRGPayload, IPEPSInteraction


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _payload() -> CTMRGPayload:
    return CTMRGPayload(
        initial_state="plus",
        interactions=[IPEPSInteraction(
            left_site=0,
            right_site=0,
            displacement=[1, 0],
            left_pauli="X",
            right_pauli="X",
            coefficient=-1.0,
        )],
        environment_bond_dim=1,
        iterations=1,
        tolerance=1e-6,
        dtype="complex64",
        optimization="full-update",
        full_update_optimizer="autodiff-ctmrg-gradient",
        optimization_steps=1,
        optimization_tolerance=1e-7,
        full_update_step=0.05,
        full_update_max_parameters=8,
        full_update_max_evaluations=8,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "optional Torch runtime is not installed")
class CTMRGAutodiffTests(unittest.TestCase):
    def test_unrolled_ctmrg_energy_has_finite_torch_gradient(self):
        import torch

        from qc_agent.core.ctmrg_autodiff import differentiable_ctmrg_energy

        tensor = torch.tensor(
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
            dtype=torch.complex64,
        ).reshape(2, 1, 1, 1, 1).requires_grad_(True)
        energy, diagnostics = differentiable_ctmrg_energy(torch, _payload(), [tensor])
        gradient = torch.autograd.grad(energy, [tensor])[0]

        self.assertAlmostEqual(float(energy.detach()), -1.0, places=5)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(math.isfinite(float(torch.linalg.norm(gradient))))
        self.assertEqual(diagnostics["gradient_backend"], "torch-autograd-unrolled-ctmrg")
        self.assertFalse(diagnostics["materializes_statevector"])

    @unittest.skipUnless(
        importlib.util.find_spec("cupy") is not None,
        "optional CuPy runtime is not installed",
    )
    def test_gpu_full_update_uses_torch_autograd_and_returns_ctmrg_result(self):
        import cupy as cp
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available to the optional Torch runtime")

        from qc_agent.core.ctmrg import run_ctmrg

        try:
            result = run_ctmrg(cp, _payload())
            diagnostics = result["optimization_diagnostics"]
            self.assertEqual(result["method"], "ipeps-full-update-autodiff-ctmrg")
            self.assertAlmostEqual(result["energy"], -1.0, places=5)
            self.assertEqual(diagnostics["optimizer"], "autodiff-ctmrg-gradient")
            self.assertEqual(diagnostics["gradient_backend"], "torch-autograd-unrolled-ctmrg")
            self.assertFalse(diagnostics["materializes_reference_statevector"])
            self.assertTrue(any("not yet an implicit fixed-point" in warning for warning in result["warnings"]))
        finally:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
