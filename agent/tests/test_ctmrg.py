import math
import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from qc_agent.core.ctmrg import run_ctmrg, run_ctmrg_convergence_study
from qc_agent.core.ctmrg_reference import finite_periodic_peps_reference
from qc_agent.core.peps import PEPSRuntime
from qc_agent.plugins.lattice import build_ctmrg_spin_payload
from qc_agent.plugins.models import CTMRGConvergenceStudyPayload, CTMRGPayload, IPEPSInteraction, LatticeHamiltonianPayload, LatticeSpec, PEPSPayload, PauliTerm


class CTMRGTests(unittest.TestCase):
    def test_environment_map_contract_is_versioned_and_projector_specific(self):
        from qc_agent.core.ctmrg_environment import environment_map_for, validate_environment_map

        baseline = environment_map_for("half-density")
        full_svd = environment_map_for("full-svd")
        bilinear = environment_map_for("biorthogonal-bilinear")
        self.assertEqual(baseline.schema, "quantum-circuit/ctmrg-environment-map-v1")
        self.assertEqual(baseline.map_id, "ctmrg-half-density-v1")
        self.assertEqual(full_svd.map_id, "ctmrg-full-svd-biorthogonal-v1")
        self.assertEqual(bilinear.map_id, "ctmrg-biorthogonal-bilinear-v1")
        self.assertEqual(baseline.virtual_leg_order, ("physical", "up", "down", "left", "right"))
        self.assertEqual(baseline.edge_order, ("T1", "T2", "T3", "T4"))
        validate_environment_map(baseline.to_dict(), baseline)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_environment_map(baseline.to_dict(), full_svd)

        payload = CTMRGPayload(
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
            environment_bond_dim=1,
            iterations=1,
        )
        result = run_ctmrg(np, payload)
        self.assertEqual(result["environment_map"]["map_id"], baseline.map_id)
        self.assertEqual(result["research_result"]["details"]["environment_map"], baseline.to_dict())

    def test_full_svd_projector_preserves_the_declared_product_limit(self):
        payload = CTMRGPayload(
            ctmrg_projector="full-svd",
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=0.5,
            )],
            environment_bond_dim=2,
            iterations=3,
        )
        result = run_ctmrg(np, payload)
        self.assertEqual(result["ctmrg_projector"], "full-svd")
        self.assertAlmostEqual(result["energy"], 1.5, places=6)
        self.assertEqual(result["research_gate"]["status"], "passed")

    def test_full_svd_entangled_gate_matches_ghz_but_remains_review_only(self):
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
            ctmrg_projector="full-svd",
            virtual_bond_dim=2,
            dtype="complex128",
            tensor_data=tensor_data,
            environment_bond_dim=2,
            iterations=4,
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
        self.assertEqual(result["ctmrg_projector"], "full-svd")
        self.assertAlmostEqual(result["energy"], 1.0, places=5)
        self.assertAlmostEqual(result["observables"][0]["value"], 0.0, places=5)
        self.assertAlmostEqual(result["interactions"][0]["value"], 1.0, places=5)
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertFalse(result["research_gate"]["production_ready"])
        self.assertEqual(result["research_gate"]["status"], "needs_review")

    def test_bilinear_projector_is_explicit_opt_in_and_keeps_ghz_reference_visible(self):
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
            ctmrg_projector="biorthogonal-bilinear",
            virtual_bond_dim=2,
            dtype="complex128",
            tensor_data=tensor_data,
            environment_bond_dim=2,
            iterations=4,
            gauge_validation=True,
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
        self.assertEqual(result["environment_map"]["map_id"], "ctmrg-biorthogonal-bilinear-v1")
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertEqual(result["research_gate"]["status"], "needs_review")
        self.assertFalse(result["research_gate"]["production_ready"])
        self.assertFalse(result["converged"])
        self.assertEqual(result["fixed_point_classification"], "unconverged")
        self.assertEqual(result["research_result"]["convergence"]["classification"], "unconverged")
        self.assertTrue(result["transported_gauge_validation"]["performed"])
        self.assertTrue(result["transported_gauge_validation"]["passed"])
        self.assertTrue(result["transported_gauge_validation"]["comparison"]["initialization_sensitive"])
        self.assertTrue(result["directional_boundary_transport_validation"]["performed"])
        self.assertTrue(result["directional_boundary_transport_validation"]["passed"])
        self.assertEqual(
            len(result["directional_boundary_transport_validation"]["directions"]),
            4,
        )
        self.assertTrue(any("paired-gauge probe fails" in warning for warning in result["warnings"]))
        self.assertTrue(any("no principled discarded-weight estimate" in warning for warning in result["warnings"]))

    def test_fixed_point_classification_distinguishes_unresolved_bilinear_gap(self):
        from qc_agent.core.ctmrg import _fixed_point_classification

        self.assertEqual(
            _fixed_point_classification(
                projector="biorthogonal-bilinear",
                converged=False,
                correlation_length=None,
            ),
            "degenerate-needs-review",
        )
        self.assertEqual(
            _fixed_point_classification(
                projector="biorthogonal-bilinear",
                converged=False,
                correlation_length=0.5,
            ),
            "unconverged",
        )

    def test_symmetry_sector_ensemble_restores_ghz_gauge_gate(self):
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
            ctmrg_projector="full-svd",
            environment_sector_policy="symmetry-ensemble",
            virtual_bond_dim=2,
            dtype="complex128",
            tensor_data=tensor_data,
            environment_bond_dim=2,
            iterations=8,
            gauge_validation=True,
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
        self.assertAlmostEqual(result["energy"], 1.0, places=5)
        self.assertAlmostEqual(result["observables"][0]["value"], 0.0, places=5)
        self.assertAlmostEqual(result["interactions"][0]["value"], 1.0, places=5)
        self.assertEqual(result["environment_sector_policy"], "symmetry-ensemble")
        self.assertEqual(result["environment_sector_count"], 2)
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertTrue(result["gauge_validation"]["passed"])
        self.assertTrue(result["research_gate"]["gates"]["virtual_gauge"]["passed"])
        self.assertFalse(result["research_gate"]["production_ready"])
        self.assertGreater(result["environment_sector_spread"]["observable_max_abs_range"], 1.0)

    def test_bilinear_symmetry_ensemble_does_not_hide_gauge_failure(self):
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
            ctmrg_projector="biorthogonal-bilinear",
            environment_sector_policy="symmetry-ensemble",
            virtual_bond_dim=2,
            dtype="complex128",
            tensor_data=tensor_data,
            environment_bond_dim=2,
            iterations=4,
            gauge_validation=True,
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
        self.assertFalse(result["converged"])
        self.assertEqual(result["fixed_point_classification"], "unconverged")
        self.assertEqual(result["research_result"]["convergence"]["classification"], "unconverged")
        self.assertFalse(result["research_gate"]["production_ready"])
        self.assertTrue(any("paired-gauge probe fails" in warning for warning in result["warnings"]))

    def test_pairwise_preconditioner_preserves_finite_reference_and_reports_candidate(self):
        from qc_agent.core.ctmrg_gauge import paired_virtual_gauge, pairwise_virtual_gauge_preconditioner

        tensor = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        tensor[0, 0, 0, 0, 0] = 1.0
        tensor[1, 1, 1, 1, 1] = 1.0
        gauged = paired_virtual_gauge(np, [tensor])[0]
        preconditioned, report = pairwise_virtual_gauge_preconditioner(np, [gauged], iterations=4)
        self.assertTrue(report["performed"])
        self.assertTrue(report["exact_periodic_pairing"])
        self.assertLess(report["vertical_pair_delta_after"], report["vertical_pair_delta_before"])
        self.assertLess(report["horizontal_pair_delta_after"], report["horizontal_pair_delta_before"])
        payload = CTMRGPayload(
            virtual_bond_dim=2,
            dtype="complex128",
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
        )
        original_reference = finite_periodic_peps_reference(
            payload, [tensor], [0.0], [1.0], 1.0, tolerance=1e-10
        )
        preconditioned_reference = finite_periodic_peps_reference(
            payload, preconditioned, [0.0], [1.0], 1.0, tolerance=1e-10
        )
        self.assertTrue(original_reference["performed"])
        self.assertTrue(preconditioned_reference["performed"])
        self.assertAlmostEqual(
            original_reference["reference_energy"],
            preconditioned_reference["reference_energy"],
            places=10,
        )

    def test_pairwise_preconditioner_rejects_condition_worsening_candidates(self):
        from qc_agent.core.ctmrg_gauge import pairwise_virtual_gauge_preconditioner

        rng = np.random.default_rng(17)
        tensor = rng.normal(size=(2, 2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2, 2))
        tensor = tensor / np.linalg.norm(tensor)
        _, report = pairwise_virtual_gauge_preconditioner(np, [tensor], iterations=4)

        self.assertEqual(report["acceptance_rule"], "bond mismatch must decrease without increasing the paired Gram condition number")
        self.assertEqual(
            report["accepted_transform_count"] + report["rejected_transform_count"],
            8,
        )
        self.assertLessEqual(
            report["condition_number_after"],
            report["condition_number_before"] * (1.0 + 1e-9),
        )

    def test_diagonal_bond_balance_reduces_periodic_metric_without_breaking_reference(self):
        from qc_agent.core.ctmrg_gauge import diagonal_bond_balance_preconditioner, paired_virtual_gauge

        tensor = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        tensor[0, 0, 0, 0, 0] = 1.0
        tensor[1, 1, 1, 1, 1] = 1.0
        gauged = paired_virtual_gauge(np, [tensor])[0]
        preconditioned, report = diagonal_bond_balance_preconditioner(
            np, [gauged], iterations=4
        )
        self.assertTrue(report["performed"])
        self.assertTrue(report["exact_periodic_pairing"])
        self.assertEqual(report["balance_power"], 0.25)
        self.assertLessEqual(
            report["vertical_pair_delta_after"],
            report["vertical_pair_delta_before"],
        )
        self.assertLessEqual(
            report["horizontal_pair_delta_after"],
            report["horizontal_pair_delta_before"],
        )
        self.assertLess(
            report["total_pair_delta_after"],
            report["total_pair_delta_before"],
        )
        payload = CTMRGPayload(
            virtual_bond_dim=2,
            dtype="complex128",
            gauge_preconditioner="diagonal-bond-balance",
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
        )
        original_reference = finite_periodic_peps_reference(
            payload, [tensor], [0.0], [1.0], 1.0, tolerance=1e-10
        )
        preconditioned_reference = finite_periodic_peps_reference(
            payload, preconditioned, [0.0], [1.0], 1.0, tolerance=1e-10
        )
        self.assertTrue(original_reference["performed"])
        self.assertTrue(preconditioned_reference["performed"])
        self.assertAlmostEqual(
            original_reference["reference_energy"],
            preconditioned_reference["reference_energy"],
            places=10,
        )

    def test_diagonal_bond_balance_is_available_through_spin_plugin_contract(self):
        payload = build_ctmrg_spin_payload(
            LatticeHamiltonianPayload(
                dimensions=[1, 1],
                model="ising",
                coupling=1.0,
                field=0.2,
            ),
            gauge_preconditioner="diagonal-bond-balance",
            gauge_preconditioner_iterations=2,
        )
        self.assertEqual(payload.gauge_preconditioner, "diagonal-bond-balance")
        bilinear_payload = build_ctmrg_spin_payload(
            LatticeHamiltonianPayload(
                dimensions=[1, 1],
                model="ising",
                coupling=1.0,
                field=0.2,
            ),
            ctmrg_projector="biorthogonal-bilinear",
        )
        self.assertEqual(bilinear_payload.ctmrg_projector, "biorthogonal-bilinear")

    def test_transport_ctm_environment_preserves_local_double_layer_contraction(self):
        from qc_agent.core.ctmrg import (
            _double_layer,
            _environment_contraction,
            _initialize_environment,
            _term_expectation,
        )
        from qc_agent.core.ctmrg_gauge import paired_virtual_gauge, transport_ctm_environment

        rng = np.random.default_rng(17)
        tensor = rng.normal(size=(2, 2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2, 2))
        tensor = (tensor / np.linalg.norm(tensor)).astype(np.complex128)
        gauged = paired_virtual_gauge(np, [tensor])[0]
        environment = _initialize_environment(np, _double_layer(np, tensor), 2)
        transported = transport_ctm_environment(np, environment, tensor)

        original_norm = _environment_contraction(
            np, environment, _double_layer(np, tensor)
        )
        transported_norm = _environment_contraction(
            np, transported, _double_layer(np, gauged)
        )
        self.assertTrue(np.allclose(original_norm, transported_norm, atol=1e-12, rtol=1e-12))

        original_z = _term_expectation(
            np,
            environment,
            tensor,
            PauliTerm(paulis={0: "Z"}, coefficient=1.0),
        )
        transported_z = _term_expectation(
            np,
            transported,
            gauged,
            PauliTerm(paulis={0: "Z"}, coefficient=1.0),
        )
        self.assertAlmostEqual(original_z, transported_z, places=10)

        result = run_ctmrg(
            np,
            CTMRGPayload(
                virtual_bond_dim=2,
                dtype="complex128",
                environment_bond_dim=2,
                iterations=2,
                gauge_validation=True,
                terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
                interactions=[IPEPSInteraction(
                    left_site=0,
                    right_site=0,
                    displacement=[1, 0],
                    left_pauli="Z",
                    right_pauli="Z",
                    coefficient=1.0,
                )],
            ),
            tensors=[tensor],
        )
        self.assertTrue(result["environment_transport_validation"]["performed"])
        self.assertTrue(result["environment_transport_validation"]["passed"])
        self.assertLess(
            result["environment_transport_validation"]["relative_max_abs_delta"],
            1e-10,
        )

    def test_boundary_basis_transport_preserves_biorthogonal_dual_rule(self):
        from qc_agent.core.ctmrg_gauge import (
            directional_boundary_gauge_map,
            paired_virtual_gauge_matrices,
            transport_bilinear_projector_pair,
            transport_biorthogonal_boundary_basis,
            transport_directional_bilinear_projector_pair,
        )

        rng = np.random.default_rng(29)
        seed = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
        projector, _ = np.linalg.qr(seed)
        enlarged_gauge = np.array([
            [1.3 + 0.1j, 0.2 - 0.2j, 0.0, 0.0],
            [0.0 + 0.1j, 0.8 - 0.05j, 0.1 + 0.2j, 0.0],
            [0.0, 0.0 + 0.2j, 1.1 - 0.1j, 0.15],
            [0.0, 0.0, 0.05 - 0.1j, 0.9 + 0.2j],
        ], dtype=np.complex128)
        left, right, report = transport_biorthogonal_boundary_basis(
            np,
            projector,
            enlarged_gauge,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(np.allclose(np.conj(left).T @ right, np.eye(2), atol=1e-10))
        self.assertTrue(
            np.allclose(
                np.conj(left).T @ enlarged_gauge,
                np.conj(projector).T,
                atol=1e-10,
            )
        )

        virtual_gauges = paired_virtual_gauge_matrices(np, np.zeros((2, 2, 2, 2, 2), dtype=np.complex128))
        for direction in ("left", "right", "top", "bottom"):
            maps = directional_boundary_gauge_map(np, virtual_gauges, 2, direction)
            self.assertEqual(maps["direction"], direction)
            self.assertEqual(maps["corner_left"].shape, (8, 8))
            self.assertEqual(maps["grown_middle"].shape, (4, 4))
            self.assertTrue(
                np.allclose(
                    maps["corner_left"].T @ maps["corner_right"],
                    np.eye(8),
                    atol=1e-10,
                )
            )
            left_probe = rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
            right_probe = rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
            _, _, directional_report = transport_directional_bilinear_projector_pair(
                np,
                left_probe,
                right_probe,
                virtual_gauges,
                boundary_dim=2,
                direction=direction,
            )
            self.assertTrue(directional_report["passed"])
            self.assertEqual(directional_report["direction"], direction)
            self.assertEqual(directional_report["factor_contract"]["middle"], "grown_middle")

    def test_directional_boundary_gauge_map_matches_all_one_site_absorptions(self):
        from qc_agent.core.ctmrg import _double_layer, _initialize_environment
        from qc_agent.core.ctmrg_gauge import (
            directional_boundary_gauge_map,
            paired_virtual_gauge,
            paired_virtual_gauge_matrices,
            transport_bilinear_projector_pair,
            transport_ctm_environment,
        )

        rng = np.random.default_rng(17)
        tensor = rng.normal(size=(2, 2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2, 2))
        tensor = (tensor / np.linalg.norm(tensor)).astype(np.complex128)
        gauged = paired_virtual_gauge(np, [tensor])[0]
        layer = _double_layer(np, tensor)
        gauged_layer = _double_layer(np, gauged)
        environment = _initialize_environment(np, layer, 2)
        gauged_environment = transport_ctm_environment(np, environment, tensor)
        virtual_gauges = paired_virtual_gauge_matrices(np, tensor)
        d2 = 4

        def factors(environment, local_layer, direction):
            if direction == "left":
                first = np.einsum("ab,buc->auc", environment.C1, environment.T1).reshape(-1, environment.T1.shape[2])
                second = np.einsum("gh,hdi->gdi", environment.C4, environment.T3).reshape(-1, environment.T3.shape[2])
                grown = np.einsum("alg,udlr->augdr", environment.T4, local_layer)
                grown = np.transpose(grown, (0, 1, 4, 2, 3)).reshape(first.shape[0], d2, second.shape[0])
            elif direction == "right":
                first = np.einsum("ce,buc->eub", environment.C2, environment.T1).reshape(-1, environment.T1.shape[0])
                second = np.einsum("im,hdi->mdh", environment.C3, environment.T3).reshape(-1, environment.T3.shape[0])
                grown = np.einsum("erm,udlr->eumdl", environment.T2, local_layer)
                grown = np.transpose(grown, (0, 1, 4, 2, 3)).reshape(first.shape[0], d2, second.shape[0])
            elif direction == "top":
                first = np.einsum("ab,alg->blg", environment.C1, environment.T4).reshape(-1, environment.T4.shape[2])
                second = np.einsum("ce,erm->crm", environment.C2, environment.T2).reshape(-1, environment.T2.shape[2])
                grown = np.einsum("buc,udlr->bcdlr", environment.T1, local_layer)
                grown = np.transpose(grown, (0, 3, 2, 1, 4)).reshape(first.shape[0], d2, second.shape[0])
            else:
                first = np.transpose(np.einsum("gh,alg->hal", environment.C4, environment.T4), (0, 2, 1)).reshape(-1, environment.T4.shape[0])
                second = np.einsum("im,erm->ire", environment.C3, environment.T2).reshape(-1, environment.T2.shape[0])
                grown = np.einsum("hdi,udlr->hiulr", environment.T3, local_layer)
                grown = np.transpose(grown, (0, 3, 2, 1, 4)).reshape(first.shape[0], d2, second.shape[0])
            return first, second, grown

        for direction in ("left", "right", "top", "bottom"):
            first, second, grown = factors(environment, layer, direction)
            gauged_first, gauged_second, gauged_grown = factors(
                gauged_environment,
                gauged_layer,
                direction,
            )
            maps = directional_boundary_gauge_map(np, virtual_gauges, 2, direction)
            predicted = np.zeros_like(grown)
            for output_middle in range(d2):
                for input_middle in range(d2):
                    predicted[:, output_middle, :] += (
                        maps["grown_middle"][output_middle, input_middle]
                        * maps["grown_row"]
                        @ grown[:, input_middle, :]
                        @ maps["grown_col"].T
                    )
            self.assertTrue(np.allclose(
                gauged_first,
                maps["corner_left"] @ first,
                atol=1e-10,
                rtol=1e-10,
            ))
            self.assertTrue(np.allclose(
                gauged_second,
                maps["corner_right"] @ second,
                atol=1e-10,
                rtol=1e-10,
            ))
            self.assertTrue(np.allclose(gauged_grown, predicted, atol=1e-10, rtol=1e-10))

        row_gauge = directional_boundary_gauge_map(
            np,
            virtual_gauges,
            2,
            "left",
        )["grown_row"]
        column_gauge = directional_boundary_gauge_map(
            np,
            virtual_gauges,
            2,
            "left",
        )["grown_col"]
        projector_rng = np.random.default_rng(41)
        left_projector = projector_rng.normal(size=(8, 2)) + 1j * projector_rng.normal(size=(8, 2))
        right_projector = projector_rng.normal(size=(8, 2)) + 1j * projector_rng.normal(size=(8, 2))
        grown_edge = projector_rng.normal(size=(8, 4, 8)) + 1j * projector_rng.normal(size=(8, 4, 8))
        gauged_edge = np.zeros_like(grown_edge)
        middle = directional_boundary_gauge_map(np, virtual_gauges, 2, "left")["grown_middle"]
        for output_middle in range(4):
            for input_middle in range(4):
                gauged_edge[:, output_middle, :] += (
                    middle[output_middle, input_middle]
                    * row_gauge
                    @ grown_edge[:, input_middle, :]
                    @ column_gauge.T
                )
        transported_left, transported_right, report = transport_bilinear_projector_pair(
            np,
            left_projector,
            right_projector,
            row_gauge,
            column_gauge,
        )
        original_projection = np.einsum(
            "ia,idj,jb->adb",
            left_projector,
            grown_edge,
            right_projector,
        )
        transported_projection = np.einsum(
            "ia,idj,jb->adb",
            transported_left,
            gauged_edge,
            transported_right,
        )
        expected_projection = np.zeros_like(original_projection)
        for output_middle in range(4):
            for input_middle in range(4):
                expected_projection[:, output_middle, :] += (
                    middle[output_middle, input_middle]
                    * original_projection[:, input_middle, :]
                )
        self.assertTrue(report["passed"])
        self.assertTrue(np.allclose(expected_projection, transported_projection, atol=1e-10, rtol=1e-10))

    def test_bond_aware_preconditioner_preserves_2x1_finite_reference(self):
        from qc_agent.core.ctmrg_gauge import pairwise_virtual_gauge_preconditioner

        rng = np.random.default_rng(29)
        tensors = [
            (rng.normal(size=(2, 2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2, 2)))
            for _ in range(2)
        ]
        tensors = [tensor / np.linalg.norm(tensor) for tensor in tensors]
        payload = CTMRGPayload(
            unit_cell=[2, 1],
            virtual_bond_dim=2,
            dtype="complex128",
            gauge_preconditioner="pairwise-polar-balance",
            interactions=[
                IPEPSInteraction(
                    left_site=0,
                    right_site=1,
                    displacement=[1, 0],
                    left_pauli="Z",
                    right_pauli="Z",
                    coefficient=0.7,
                ),
                IPEPSInteraction(
                    left_site=1,
                    right_site=0,
                    displacement=[1, 0],
                    left_pauli="X",
                    right_pauli="X",
                    coefficient=-0.2,
                ),
            ],
        )
        preconditioned, report = pairwise_virtual_gauge_preconditioner(
            np, tensors, unit_cell=(2, 1), iterations=2
        )
        self.assertTrue(report["performed"])
        self.assertEqual(len(report["bond_metrics"]), 4)
        original_reference = finite_periodic_peps_reference(
            payload, tensors, [], [0.0, 0.0], 0.0, tolerance=1e-8
        )
        preconditioned_reference = finite_periodic_peps_reference(
            payload, preconditioned, [], [0.0, 0.0], 0.0, tolerance=1e-8
        )
        self.assertTrue(original_reference["performed"])
        self.assertTrue(preconditioned_reference["performed"])
        self.assertAlmostEqual(
            original_reference["reference_energy"],
            preconditioned_reference["reference_energy"],
            places=8,
        )

    def test_bond_aware_preconditioner_preserves_2x2_finite_reference(self):
        from qc_agent.core.ctmrg_gauge import pairwise_virtual_gauge_preconditioner

        rng = np.random.default_rng(41)
        tensors = [
            (rng.normal(size=(2, 2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2, 2)))
            for _ in range(4)
        ]
        tensors = [tensor / np.linalg.norm(tensor) for tensor in tensors]
        interactions = [
            IPEPSInteraction(left_site=0, right_site=1, displacement=[1, 0], left_pauli="Z", right_pauli="Z", coefficient=0.4),
            IPEPSInteraction(left_site=0, right_site=2, displacement=[0, 1], left_pauli="X", right_pauli="X", coefficient=-0.3),
            IPEPSInteraction(left_site=1, right_site=0, displacement=[1, 0], left_pauli="Y", right_pauli="Y", coefficient=0.2),
            IPEPSInteraction(left_site=2, right_site=0, displacement=[0, 1], left_pauli="Z", right_pauli="Z", coefficient=0.1),
        ]
        payload = CTMRGPayload(
            unit_cell=[2, 2],
            virtual_bond_dim=2,
            dtype="complex128",
            gauge_preconditioner="pairwise-polar-balance",
            interactions=interactions,
        )
        preconditioned, report = pairwise_virtual_gauge_preconditioner(
            np, tensors, unit_cell=(2, 2), iterations=2
        )
        self.assertTrue(report["performed"])
        self.assertEqual(len(report["bond_metrics"]), 8)
        original_reference = finite_periodic_peps_reference(
            payload, tensors, [], [0.0] * 4, 0.0, tolerance=1e-8
        )
        preconditioned_reference = finite_periodic_peps_reference(
            payload, preconditioned, [], [0.0] * 4, 0.0, tolerance=1e-8
        )
        self.assertTrue(original_reference["performed"])
        self.assertTrue(preconditioned_reference["performed"])
        self.assertAlmostEqual(
            original_reference["reference_energy"],
            preconditioned_reference["reference_energy"],
            places=8,
        )

    def test_full_svd_projector_rejects_unbounded_virtual_bond(self):
        with self.assertRaisesRegex(ValueError, "full-svd CTMRG projectors"):
            CTMRGPayload(
                ctmrg_projector="full-svd",
                virtual_bond_dim=3,
                interactions=[IPEPSInteraction(
                    left_site=0,
                    right_site=0,
                    displacement=[1, 0],
                    left_pauli="Z",
                    right_pauli="Z",
                    coefficient=1.0,
                )],
            )

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
        self.assertEqual(result["research_gate"]["status"], "passed")
        self.assertTrue(result["research_gate"]["production_ready"])

    def test_environment_under_relaxation_preserves_product_reference(self):
        result = run_ctmrg(np, CTMRGPayload(
            environment_damping=0.25,
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
                coefficient=0.5,
            )],
        ))
        self.assertEqual(result["environment_damping"], 0.25)
        self.assertAlmostEqual(result["energy"], 1.5, places=6)
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertTrue(any("under-relaxation damping" in warning for warning in result["warnings"]))

    def test_boundary_mps_reference_preserves_product_limit_and_surfaces_diagnostics(self):
        result = run_ctmrg(np, CTMRGPayload(
            boundary_mps_reference=True,
            boundary_mps_width=3,
            boundary_mps_height=3,
            boundary_mps_bond_dim=4,
            environment_bond_dim=2,
            iterations=3,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=0.5,
            )],
        ))
        reference = result["boundary_mps_reference"]
        self.assertTrue(reference["performed"])
        self.assertEqual(reference["reference"], "finite-cylinder-boundary-mps")
        self.assertAlmostEqual(reference["reference_energy"], 1.5, places=6)
        self.assertAlmostEqual(reference["max_abs_error"], 0.0, places=8)
        self.assertEqual(reference["diagnostics"]["norm_contraction"]["boundary_bond_dim_used"], 1)
        self.assertTrue(any("finite-cylinder boundary-MPS reference" in warning for warning in result["warnings"]))

    def test_boundary_mps_convergence_study_is_bounded_and_replayable(self):
        from qc_agent.core.ctmrg_boundary_mps import run_boundary_mps_convergence_study

        tensor = np.zeros((2, 1, 1, 1, 1), dtype=np.complex128)
        tensor[0, 0, 0, 0, 0] = 1.0
        payload = CTMRGPayload(
            boundary_mps_width=2,
            boundary_mps_height=2,
            boundary_mps_bond_dim=1,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=0.5,
            )],
        )
        study = run_boundary_mps_convergence_study(
            [tensor],
            payload,
            ctmrg_energy=1.5,
            ctmrg_onsite=[1.0],
            ctmrg_interactions=[1.0],
            patch_sizes=[(2, 2), (3, 3)],
            boundary_bond_dims=[1, 2],
        )
        self.assertEqual(study["point_count"], 4)
        self.assertTrue(all(point["performed"] for point in study["points"]))
        self.assertTrue(all(abs(point["reference_energy"] - 1.5) < 1e-8 for point in study["points"]))
        self.assertEqual(study["max_discarded_weight"], 0.0)

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
        self.assertGreater(result["raw_boundary_basis_residual"], result["residual"])
        self.assertTrue(result["reference_validation"]["performed"])
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertEqual(result["reference_validation"]["reference"], "analytic-ghz-transfer-fixed-point")
        self.assertEqual(result["research_gate"]["status"], "needs_review")
        self.assertFalse(result["research_gate"]["production_ready"])
        self.assertFalse(result["research_gate"]["gates"]["virtual_gauge"]["passed"])

    def test_generic_virtual_two_tensor_gets_finite_periodic_reference(self):
        tensor = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        tensor[0, 0, 0, 0, 0] = 1.0
        tensor[0, 1, 1, 1, 1] = 0.5
        tensor[1, 0, 0, 0, 0] = 0.3
        tensor[1, 1, 1, 1, 1] = 0.2
        result = run_ctmrg(np, CTMRGPayload(
            unit_cell=[1, 1],
            virtual_bond_dim=2,
            dtype="complex128",
            tensor_data=[[float(value.real), float(value.imag)] for value in tensor.reshape(-1)],
            environment_bond_dim=2,
            iterations=3,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=0.2)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=0.4,
            )],
        ))
        reference = result["reference_validation"]
        self.assertTrue(reference["performed"])
        self.assertEqual(reference["reference"], "finite-periodic-peps-2x2")
        self.assertEqual(reference["reference_sites"], 4)
        self.assertTrue(math.isfinite(reference["reference_energy"]))
        self.assertFalse(reference["passed"])
        self.assertTrue(any("finite-periodic-peps-2x2" in warning for warning in result["warnings"]))
        self.assertTrue(any("finite torus" in limitation for limitation in result["research_result"]["limitations"]))

    def test_generic_two_by_two_unit_cell_gets_independent_finite_reference(self):
        tensors: list[list[float]] = []
        for site in range(4):
            tensor = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
            tensor[0, 0, 0, 0, 0] = 1.0 + 0.05j * site
            tensor[1, 1, 1, 1, 1] = 0.2 + 0.03j * site
            tensor[0, 0, 1, 1, 0] = 0.1
            tensors.extend([[float(value.real), float(value.imag)] for value in tensor.reshape(-1)])
        result = run_ctmrg(np, CTMRGPayload(
            unit_cell=[2, 2],
            virtual_bond_dim=2,
            dtype="complex128",
            tensor_data=tensors,
            environment_bond_dim=2,
            iterations=3,
            terms=[PauliTerm(paulis={3: "Z"}, coefficient=0.1)],
            interactions=[
                IPEPSInteraction(
                    left_site=0,
                    right_site=1,
                    displacement=[1, 0],
                    left_pauli="Z",
                    right_pauli="Z",
                    coefficient=0.2,
                ),
                IPEPSInteraction(
                    left_site=0,
                    right_site=2,
                    displacement=[0, 1],
                    left_pauli="X",
                    right_pauli="X",
                    coefficient=-0.1,
                ),
            ],
        ))
        reference = result["reference_validation"]
        self.assertTrue(reference["performed"])
        self.assertEqual(reference["reference"], "finite-periodic-peps-2x2")
        self.assertEqual(reference["reference_unit_cell"], [2, 2])
        self.assertEqual(reference["reference_lattice"], [2, 2])
        self.assertEqual(reference["reference_sites"], 4)
        self.assertTrue(math.isfinite(reference["reference_energy"]))
        self.assertFalse(reference["passed"])

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
            self.assertEqual(
                partial["checkpoint"]["metadata"]["environment_map"]["map_id"],
                "ctmrg-half-density-v1",
            )
            resumed = run_ctmrg(np, payload.model_copy(update={
                "iterations": 4,
                "resume_from": checkpoint_path,
                "checkpoint_path": checkpoint_path,
            }))
            fresh = run_ctmrg(np, payload.model_copy(update={"iterations": 4}))
            self.assertEqual(resumed["unit_cell"], [2, 2])
            self.assertEqual(resumed["initial_environment_source"], "checkpoint")
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
            full_update_spsa_directions=4,
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

    def test_finite_torus_gradient_full_update_is_explicit_and_reference_checked(self):
        result = run_ctmrg(np, CTMRGPayload(
            unit_cell=[1, 1],
            virtual_bond_dim=2,
            dtype="complex128",
            initial_state="plus",
            optimization="full-update",
            full_update_optimizer="finite-torus-gradient",
            optimization_steps=6,
            full_update_step=0.1,
            full_update_max_parameters=128,
            full_update_max_evaluations=64,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=-1.0)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=0.0,
            )],
            environment_bond_dim=2,
            iterations=3,
        ))
        diagnostics = result["optimization_diagnostics"]
        self.assertEqual(result["method"], "ipeps-full-update-finite-torus-gradient-ctmrg")
        self.assertEqual(diagnostics["optimizer"], "finite-torus-gradient")
        self.assertEqual(diagnostics["gradient_backend"], "analytic-finite-torus-reverse-contraction")
        self.assertLess(diagnostics["final_energy"], -3.9)
        self.assertTrue(diagnostics["materializes_reference_statevector"])
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertTrue(any("finite-torus-gradient" in warning for warning in result["warnings"]))

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
        self.assertEqual(diagnostics["direction_count"], 4)
        self.assertLessEqual(diagnostics["evaluations"], 1 + 2 * 12)
        self.assertGreater(len(set(round(value, 10) for value in diagnostics["energy_history"][0]["gradient_scales"])), 1)
        self.assertTrue(any("SPSA" in warning for warning in result["warnings"]))

    def test_spsa_reaches_known_d1_product_limit_without_statevector(self):
        result = run_ctmrg(np, CTMRGPayload(
            dtype="complex128",
            initial_state="plus",
            optimization="full-update",
            full_update_optimizer="spsa-gradient",
            full_update_spsa_directions=1,
            optimization_steps=16,
            full_update_step=0.05,
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
        self.assertLess(diagnostics["final_energy"], -1.19)
        self.assertLess(diagnostics["evaluations"], 128)
        self.assertTrue(result["reference_validation"]["passed"])
        self.assertFalse(diagnostics["evaluation_budget_exhausted"])

    def test_spsa_optimizer_checkpoint_resume_matches_fresh_run(self):
        common = {
            "dtype": "complex128",
            "initial_state": "plus",
            "optimization": "full-update",
            "full_update_optimizer": "spsa-gradient",
            "full_update_spsa_directions": 1,
            "full_update_step": 0.05,
            "full_update_gradient_epsilon": 1e-3,
            "full_update_max_parameters": 8,
            "full_update_max_evaluations": 64,
            "terms": [PauliTerm(paulis={0: "Z"}, coefficient=-0.2)],
            "interactions": [IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=-1.0,
            )],
            "environment_bond_dim": 1,
            "iterations": 1,
        }
        with TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "spsa-state.npz")
            partial = run_ctmrg(np, CTMRGPayload(**common, optimization_steps=2, optimizer_checkpoint_path=checkpoint))
            resumed = run_ctmrg(np, CTMRGPayload(
                **common,
                optimization_steps=6,
                optimizer_checkpoint_path=checkpoint,
                optimizer_resume_from=checkpoint,
            ))
            fresh = run_ctmrg(np, CTMRGPayload(**common, optimization_steps=6))

            partial_diagnostics = partial["optimization_diagnostics"]
            resumed_diagnostics = resumed["optimization_diagnostics"]
            self.assertTrue(partial_diagnostics["checkpoint"]["resumable"])
            self.assertEqual(partial_diagnostics["checkpoint"]["step"], 2)
            self.assertEqual(resumed_diagnostics["start_iteration"], 2)
            self.assertAlmostEqual(resumed_diagnostics["final_energy"], fresh["optimization_diagnostics"]["final_energy"], places=12)
            self.assertAlmostEqual(resumed["energy"], fresh["energy"], places=12)

    def test_bounded_feedback_optimizer_checkpoints_resume_for_coordinate_and_finite_difference(self):
        common = {
            "dtype": "complex128",
            "initial_state": "plus",
            "optimization": "full-update",
            "full_update_step": 0.05,
            "full_update_gradient_epsilon": 1e-3,
            "full_update_max_parameters": 8,
            "full_update_max_evaluations": 128,
            "terms": [PauliTerm(paulis={0: "Z"}, coefficient=-0.2)],
            "interactions": [IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=-1.0,
            )],
            "environment_bond_dim": 1,
            "iterations": 1,
        }
        for optimizer in ("coordinate", "finite-difference-gradient"):
            with self.subTest(optimizer=optimizer), TemporaryDirectory() as directory:
                checkpoint = os.path.join(directory, f"{optimizer}.npz")
                partial = run_ctmrg(np, CTMRGPayload(
                    **common,
                    full_update_optimizer=optimizer,
                    optimization_steps=1,
                    optimizer_checkpoint_path=checkpoint,
                ))
                resumed = run_ctmrg(np, CTMRGPayload(
                    **common,
                    full_update_optimizer=optimizer,
                    optimization_steps=2,
                    optimizer_checkpoint_path=checkpoint,
                    optimizer_resume_from=checkpoint,
                ))
                fresh = run_ctmrg(np, CTMRGPayload(
                    **common,
                    full_update_optimizer=optimizer,
                    optimization_steps=2,
                ))
                self.assertTrue(partial["optimization_diagnostics"]["checkpoint"]["resumable"])
                self.assertEqual(resumed["optimization_diagnostics"]["start_iteration"], 1)
                self.assertAlmostEqual(
                    resumed["optimization_diagnostics"]["final_energy"],
                    fresh["optimization_diagnostics"]["final_energy"],
                    places=10,
                )
                self.assertAlmostEqual(resumed["energy"], fresh["energy"], places=10)

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
            self.assertEqual(partial["checkpoint"]["metadata"]["environment_map"]["schema"], "quantum-circuit/ctmrg-environment-map-v1")
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

    def test_contraction_checkpoint_resume_is_rejected_for_optimizer_runs(self):
        with self.assertRaisesRegex(ValueError, "optimizer checkpoint/resume"):
            CTMRGPayload(
                optimization="full-update",
                checkpoint_path="optimizer.npz",
                interactions=[{
                    "left_site": 0,
                    "right_site": 0,
                    "displacement": [1, 0],
                    "left_pauli": "Z",
                    "right_pauli": "Z",
                    "coefficient": 1.0,
                }],
            )

    def test_optimizer_state_checkpoint_is_limited_to_supported_strategies(self):
        with self.assertRaisesRegex(ValueError, "optimizer checkpoint state"):
            CTMRGPayload(
                optimization="simple-update",
                full_update_optimizer="coordinate",
                optimizer_checkpoint_path="optimizer.npz",
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
        self.assertIn("gauge_conditioning", study)
        self.assertTrue(study["gauge_conditioning"]["performed"])
        self.assertTrue(all("raw_boundary_basis_residual" in point for point in study["points"]))
        self.assertEqual([point["environment_bond_dim"] for point in study["points"]], [1, 2, 4])
        self.assertEqual(study["reference_summary"]["consistent_reference"], "finite-product-supercell")
        self.assertEqual(study["reference_summary"]["performed_points"], 3)
        self.assertEqual(study["reference_summary"]["passed_points"], 3)
        self.assertEqual(study["environment_sector_summary"]["policies"], ["single"])
        self.assertEqual(study["environment_sector_summary"]["sector_counts"], [1])
        self.assertEqual(study["environment_sector_summary"]["max_spread"]["energy_abs_range"], 0.0)
        self.assertEqual(study["boundary_mps_summary"]["performed_points"], 0)
        self.assertIn("research_gate_summary", study)
        self.assertEqual(study["research_gate_summary"]["status"], "passed")
        self.assertTrue(study["research_gate_summary"]["production_ready"])
        self.assertEqual(study["research_gate_summary"]["passed_points"], 3)
        self.assertTrue(all(point["research_gate_status"] == "passed" for point in study["points"]))
        self.assertTrue(all(point["research_gate_production_ready"] for point in study["points"]))
        self.assertFalse(study["materializes_statevector"])
        self.assertIsNone(study["points"][0]["energy_delta"])
        self.assertIsNone(study["points"][0]["observable_max_abs_delta"])
        for point in study["points"]:
            self.assertTrue(math.isfinite(point["energy"]))
            self.assertTrue(math.isfinite(point["residual"]))
            self.assertTrue(math.isfinite(point["correlation_length"]))
            self.assertGreaterEqual(len(point["environment_spectrum"]), 1)
            self.assertEqual(point["reference_name"], "finite-product-supercell")
            self.assertTrue(point["reference_passed"])

    def test_environment_dimension_study_preserves_symmetry_sector_spread(self):
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
        study = run_ctmrg_convergence_study(np, CTMRGPayload(
            ctmrg_projector="full-svd",
            environment_sector_policy="symmetry-ensemble",
            virtual_bond_dim=2,
            dtype="complex128",
            tensor_data=tensor_data,
            environment_bond_dim=2,
            iterations=3,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            interactions=[IPEPSInteraction(
                left_site=0,
                right_site=0,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=1.0,
            )],
        ), [1, 2])
        self.assertEqual(study["environment_sector_summary"]["policies"], ["symmetry-ensemble"])
        self.assertEqual(study["environment_sector_summary"]["sector_counts"], [2])
        self.assertGreater(
            study["environment_sector_summary"]["max_spread"]["observable_max_abs_range"],
            1.0,
        )
        self.assertTrue(all(point["environment_sector_spread"] for point in study["points"]))

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
