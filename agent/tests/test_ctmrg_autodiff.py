import importlib.util
import math
import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from qc_agent.core.ctmrg_reference import finite_periodic_peps_reference
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

    def test_implicit_fixed_point_gradient_matches_central_difference(self):
        import torch

        from qc_agent.core.ctmrg_autodiff import implicit_ctmrg_energy_and_gradient

        payload = _payload().model_copy(update={
            "dtype": "complex128",
            "iterations": 3,
            "interactions": [IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="X",
                right_pauli="X",
                coefficient=-0.7,
            )],
            "full_update_implicit_iterations": 8,
            "full_update_implicit_tolerance": 1e-8,
        })
        base = torch.tensor(
            [0.83 + 0.11j, 0.51 - 0.07j],
            dtype=torch.complex128,
        ).reshape(2, 1, 1, 1, 1).requires_grad_(True)
        energy, gradients, diagnostics = implicit_ctmrg_energy_and_gradient(torch, payload, [base])
        gradient = gradients[0].detach().reshape(-1)
        finite_difference: list[float] = []
        epsilon = 1e-6
        for index in range(base.numel()):
            for direction in (1.0, 1.0j):
                plus = base.detach().clone()
                minus = base.detach().clone()
                plus.reshape(-1)[index] += epsilon * direction
                minus.reshape(-1)[index] -= epsilon * direction
                plus_energy, _, _ = implicit_ctmrg_energy_and_gradient(
                    torch, payload, [plus.requires_grad_(True)]
                )
                minus_energy, _, _ = implicit_ctmrg_energy_and_gradient(
                    torch, payload, [minus.requires_grad_(True)]
                )
                finite_difference.append(float((plus_energy - minus_energy) / (2.0 * epsilon)))

        autodiff_components = [
            float(gradient[0].real),
            float(gradient[0].imag),
            float(gradient[1].real),
            float(gradient[1].imag),
        ]
        np.testing.assert_allclose(autodiff_components, finite_difference, rtol=2e-5, atol=2e-6)
        self.assertLessEqual(diagnostics["adjoint_residual"], 1e-8)
        self.assertGreater(diagnostics["transfer_gap"], 0.0)
        self.assertTrue(math.isfinite(float(energy.detach())))

    def test_independent_finite_peps_reference_preserves_virtual_gauge(self):
        payload = _payload().model_copy(update={
            "dtype": "complex128",
            "virtual_bond_dim": 2,
            "environment_bond_dim": 2,
        })
        generator = np.random.default_rng(17)
        tensor = generator.normal(size=(2, 2, 2, 2, 2)) + 1j * generator.normal(size=(2, 2, 2, 2, 2))
        tensor = tensor / np.linalg.norm(tensor)
        gauge = np.asarray(
            [[1.2 + 0.1j, 0.2 - 0.1j], [0.0 + 0.2j, 0.8 - 0.05j]],
            dtype=np.complex128,
        )
        inverse_transpose = np.linalg.inv(gauge).T
        transformed = np.einsum(
            "sUDLR,uU,dD,lL,rR->sudlr",
            tensor,
            inverse_transpose,
            gauge,
            inverse_transpose,
            gauge,
        )
        original = finite_periodic_peps_reference(payload, [tensor], [], [0.0], 0.0)
        gauged = finite_periodic_peps_reference(payload, [transformed], [], [0.0], 0.0)
        self.assertTrue(original["performed"])
        self.assertTrue(gauged["performed"])
        self.assertAlmostEqual(original["reference_energy"], gauged["reference_energy"], places=10)

    def test_ctmrg_gauge_probe_surfaces_truncation_sensitivity(self):
        from qc_agent.core.ctmrg import run_ctmrg
        from qc_agent.core.ctmrg_gauge import virtual_leg_conditioning_report

        rng = np.random.default_rng(17)
        tensor = rng.normal(size=(2, 2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2, 2))
        tensor = tensor / np.linalg.norm(tensor)
        payload = _payload().model_copy(update={
            "optimization": "none",
            "dtype": "complex128",
            "virtual_bond_dim": 2,
            "tensor_data": [[float(value.real), float(value.imag)] for value in tensor.reshape(-1)],
            "environment_bond_dim": 2,
            "iterations": 4,
            "gauge_validation": True,
        })
        result = run_ctmrg(np, payload)
        gauge = result["gauge_validation"]
        self.assertTrue(gauge["performed"])
        self.assertFalse(gauge["passed"])
        self.assertGreater(gauge["max_abs_delta"], gauge["tolerance"])
        self.assertTrue(any("virtual-gauge validation" in warning for warning in result["warnings"]))
        self.assertTrue(result["gauge_conditioning"]["performed"])
        self.assertEqual(len(result["gauge_conditioning"]["legs"]), 4)

        rank_deficient = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        rank_deficient[0, 0, 0, 0, 0] = 1.0
        before = rank_deficient.copy()
        conditioning = virtual_leg_conditioning_report(np, [rank_deficient])
        self.assertFalse(conditioning["well_conditioned"])
        self.assertEqual(conditioning["legs"][0]["rank_estimate"], 1)
        np.testing.assert_array_equal(rank_deficient, before)

    def test_unrolled_optimizer_checkpoint_resume_matches_fresh_cpu_run(self):
        from qc_agent.core.ctmrg import run_ctmrg

        common = _payload().model_copy(update={
            "dtype": "complex128",
            "optimization_steps": 2,
            "full_update_max_evaluations": 64,
        })
        with TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "autodiff-state.npz")
            partial = run_ctmrg(np, common.model_copy(update={
                "optimization_steps": 1,
                "optimizer_checkpoint_path": checkpoint,
            }))
            resumed = run_ctmrg(np, common.model_copy(update={
                "optimizer_checkpoint_path": checkpoint,
                "optimizer_resume_from": checkpoint,
            }))
            fresh = run_ctmrg(np, common)

            self.assertTrue(partial["optimization_diagnostics"]["checkpoint"]["resumable"])
            self.assertEqual(partial["optimization_diagnostics"]["checkpoint"]["step"], 1)
            self.assertEqual(resumed["optimization_diagnostics"]["start_iteration"], 1)
            self.assertAlmostEqual(
                resumed["optimization_diagnostics"]["final_energy"],
                fresh["optimization_diagnostics"]["final_energy"],
                places=12,
            )
            self.assertAlmostEqual(resumed["energy"], fresh["energy"], places=12)

    def test_implicit_optimizer_checkpoint_resume_matches_fresh_cpu_run(self):
        from qc_agent.core.ctmrg import run_ctmrg

        common = _payload().model_copy(update={
            "dtype": "complex128",
            "full_update_optimizer": "implicit-ctmrg-gradient",
            "optimization_steps": 2,
            "iterations": 2,
            "full_update_implicit_iterations": 8,
            "full_update_implicit_tolerance": 1e-8,
            "full_update_max_evaluations": 64,
        })
        with TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "implicit-state.npz")
            partial = run_ctmrg(np, common.model_copy(update={
                "optimization_steps": 1,
                "optimizer_checkpoint_path": checkpoint,
            }))
            resumed = run_ctmrg(np, common.model_copy(update={
                "optimizer_checkpoint_path": checkpoint,
                "optimizer_resume_from": checkpoint,
            }))
            fresh = run_ctmrg(np, common)

            self.assertTrue(partial["optimization_diagnostics"]["checkpoint"]["resumable"])
            self.assertEqual(partial["optimization_diagnostics"]["checkpoint"]["step"], 1)
            self.assertEqual(resumed["optimization_diagnostics"]["start_iteration"], 1)
            self.assertAlmostEqual(
                resumed["optimization_diagnostics"]["final_energy"],
                fresh["optimization_diagnostics"]["final_energy"],
                places=10,
            )
            self.assertAlmostEqual(resumed["energy"], fresh["energy"], places=10)

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

    @unittest.skipUnless(
        importlib.util.find_spec("cupy") is not None,
        "optional CuPy runtime is not installed",
    )
    def test_gpu_implicit_full_update_reports_adjoint_diagnostics(self):
        import cupy as cp
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available to the optional Torch runtime")

        from qc_agent.core.ctmrg import run_ctmrg

        payload = _payload().model_copy(update={
            "optimization": "full-update",
            "full_update_optimizer": "implicit-ctmrg-gradient",
            "iterations": 2,
            "full_update_implicit_iterations": 8,
            "full_update_implicit_tolerance": 1e-5,
        })
        try:
            result = run_ctmrg(cp, payload)
            diagnostics = result["optimization_diagnostics"]
            self.assertEqual(result["method"], "ipeps-full-update-implicit-ctmrg")
            self.assertAlmostEqual(result["energy"], -1.0, places=5)
            self.assertEqual(diagnostics["optimizer"], "implicit-ctmrg-gradient")
            self.assertEqual(diagnostics["gradient_backend"], "torch-autograd-implicit-fixed-point")
            self.assertGreaterEqual(diagnostics["transfer_gap"], 0.0)
            self.assertTrue(math.isfinite(float(diagnostics["adjoint_residual"])))
        finally:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

    @unittest.skipUnless(
        importlib.util.find_spec("cupy") is not None,
        "optional CuPy runtime is not installed",
    )
    def test_gpu_unrolled_optimizer_checkpoint_round_trip(self):
        import cupy as cp
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available to the optional Torch runtime")

        from qc_agent.core.ctmrg import run_ctmrg

        common = _payload().model_copy(update={
            "optimization_steps": 2,
            "full_update_max_evaluations": 64,
        })
        try:
            with TemporaryDirectory() as directory:
                checkpoint = os.path.join(directory, "gpu-autodiff-state.npz")
                partial = run_ctmrg(cp, common.model_copy(update={
                    "optimization_steps": 1,
                    "optimizer_checkpoint_path": checkpoint,
                }))
                resumed = run_ctmrg(cp, common.model_copy(update={
                    "optimizer_checkpoint_path": checkpoint,
                    "optimizer_resume_from": checkpoint,
                }))
                fresh = run_ctmrg(cp, common)
                self.assertTrue(partial["optimization_diagnostics"]["checkpoint"]["resumable"])
                self.assertEqual(resumed["optimization_diagnostics"]["start_iteration"], 1)
                self.assertAlmostEqual(
                    resumed["optimization_diagnostics"]["final_energy"],
                    fresh["optimization_diagnostics"]["final_energy"],
                    places=5,
                )
        finally:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

    @unittest.skipUnless(
        importlib.util.find_spec("cupy") is not None,
        "optional CuPy runtime is not installed",
    )
    def test_gpu_implicit_optimizer_checkpoint_round_trip(self):
        import cupy as cp
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available to the optional Torch runtime")

        from qc_agent.core.ctmrg import run_ctmrg

        common = _payload().model_copy(update={
            "full_update_optimizer": "implicit-ctmrg-gradient",
            "optimization_steps": 2,
            "iterations": 2,
            "full_update_implicit_iterations": 8,
            "full_update_implicit_tolerance": 1e-5,
            "full_update_max_evaluations": 64,
        })
        try:
            with TemporaryDirectory() as directory:
                checkpoint = os.path.join(directory, "gpu-implicit-state.npz")
                partial = run_ctmrg(cp, common.model_copy(update={
                    "optimization_steps": 1,
                    "optimizer_checkpoint_path": checkpoint,
                }))
                resumed = run_ctmrg(cp, common.model_copy(update={
                    "optimizer_checkpoint_path": checkpoint,
                    "optimizer_resume_from": checkpoint,
                }))
                fresh = run_ctmrg(cp, common)
                self.assertTrue(partial["optimization_diagnostics"]["checkpoint"]["resumable"])
                self.assertEqual(resumed["optimization_diagnostics"]["start_iteration"], 1)
                self.assertAlmostEqual(
                    resumed["optimization_diagnostics"]["final_energy"],
                    fresh["optimization_diagnostics"]["final_energy"],
                    places=5,
                )
        finally:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
