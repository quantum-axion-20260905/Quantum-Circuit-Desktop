import unittest
import math
from tempfile import TemporaryDirectory

import numpy as np

from qc_agent.core.mps_runtime import MPSRuntime
from qc_agent.core.boundary_mps import contract_boundary_mps
from qc_agent.core.peps import PEPSRuntime
from qc_agent.core.ground_state import exact_ground_state
from qc_agent.core.dmrg import run_dmrg
from qc_agent.core.peps import run_peps
from qc_agent.backends.preflight import estimate_dmrg, estimate_peps, estimate_tebd
from qc_agent.models import TNGate, TNPayload
from qc_agent.plugins.lattice import build_spin_hamiltonian, lattice_graph
from qc_agent.plugins.fermion import map_fermion_terms
from qc_agent.plugins.materials import build_hubbard_hamiltonian
from qc_agent.plugins.models import (
    ExpectationPayload,
    FermionMappingPayload,
    FermionOperator,
    FermionTerm,
    GroundStatePayload,
    DMRGPayload,
    HubbardPayload,
    LatticeHamiltonianPayload,
    PauliTerm,
    PEPSPayload,
    TEBDPayload,
)
from qc_agent.plugins.tebd import run_tebd


class PhysicsPluginTests(unittest.TestCase):
    def test_two_dimensional_snake_lattice_and_ising_terms(self):
        spec = LatticeHamiltonianPayload(dimensions=[3, 2], model="ising", coupling=1.0, field=0.5)
        graph = lattice_graph(spec)
        hamiltonian = build_spin_hamiltonian(spec)
        self.assertEqual(len(graph["sites"]), 6)
        self.assertEqual(len(graph["edges"]), 7)
        self.assertEqual(len(hamiltonian["terms"]), 13)

    def test_periodic_single_site_axis_does_not_create_self_edge(self):
        graph = lattice_graph(LatticeHamiltonianPayload(dimensions=[1, 2], boundary="periodic"))
        self.assertTrue(all(edge["source"] != edge["target"] for edge in graph["edges"]))

    def test_jordan_wigner_maps_number_operator_to_real_paulis(self):
        payload = FermionMappingPayload(
            n_modes=1,
            terms=[FermionTerm(
                operators=[
                    FermionOperator(mode=0, action="create"),
                    FermionOperator(mode=0, action="annihilate"),
                ],
                coefficient=1.0,
                label="n0",
            )],
        )
        result = map_fermion_terms(payload)
        self.assertTrue(result["expectation_ready"])
        self.assertEqual(result["terms"], [
            {"paulis": {}, "coefficient": 0.5, "label": "I"},
            {"paulis": {"0": "Z"}, "coefficient": -0.5, "label": "Z0"},
        ])

    def test_hubbard_plugin_returns_expectation_ready_2d_contract(self):
        result = build_hubbard_hamiltonian(HubbardPayload(
            dimensions=[2, 2],
            hopping=0.7,
            onsite_u=1.2,
            chemical_potential=0.1,
        ))
        self.assertEqual(result["n_qubits"], 8)
        self.assertTrue(result["expectation_ready"])
        self.assertTrue(result["tebd_ready"])
        self.assertGreater(result["max_pauli_locality"], 2)
        self.assertGreater(len(result["fermion_terms"]), 0)
        self.assertGreater(len(result["terms"]), 0)

    def test_exact_ground_state_reference_solves_single_qubit_pauli(self):
        payload = GroundStatePayload(
            n_qubits=1,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            bitstrings=["0", "1"],
        )
        result = exact_ground_state(np, payload)
        self.assertEqual(result["backend"], "exact-diagonalization-reference")
        self.assertAlmostEqual(result["ground_energy"], -1.0, places=6)
        self.assertAlmostEqual(result["norm2"], 1.0, places=6)

    def test_two_site_dmrg_finds_product_ground_state(self):
        payload = DMRGPayload(
            n_qubits=2,
            terms=[
                PauliTerm(paulis={0: "Z"}, coefficient=1.0),
                PauliTerm(paulis={1: "Z"}, coefficient=1.0),
            ],
            bond_dim=2,
            sweeps=3,
        )
        result = run_dmrg(np, payload)
        self.assertAlmostEqual(result["ground_energy"], -2.0, places=5)
        self.assertAlmostEqual(result["norm2"], 1.0, places=5)
        self.assertEqual(result["backend"], "tensor-network-mps-dmrg")
        self.assertIn("energy_variance", result)
        self.assertIn("truncation_report", result)
        self.assertIn("resource_estimate", result)
        self.assertTrue(result["exact_cross_check"]["performed"])
        self.assertTrue(result["exact_cross_check"]["passed"])
        self.assertEqual(result["research_result"]["schema"], "quantum-circuit/research-result-v1")
        self.assertEqual(result["research_result"]["representation"], "mps")

    def test_two_site_dmrg_matches_entangled_heisenberg_reference(self):
        terms = [
            PauliTerm(paulis={0: pauli, 1: pauli}, coefficient=1.0)
            for pauli in ("X", "Y", "Z")
        ]
        result = run_dmrg(np, DMRGPayload(
            n_qubits=2,
            terms=terms,
            bond_dim=2,
            sweeps=4,
        ))
        exact = exact_ground_state(np, GroundStatePayload(n_qubits=2, terms=terms))
        self.assertAlmostEqual(result["ground_energy"], exact["ground_energy"], places=5)
        self.assertTrue(result["converged"])
        self.assertLess(result["local_solver_residual"], 1e-5)

    def test_spin_chain_dmrg_models_match_small_exact_references(self):
        for model, anisotropy in (("ising", 1.0), ("heisenberg", 1.0), ("xxz", 0.7)):
            with self.subTest(model=model):
                spec = LatticeHamiltonianPayload(
                    dimensions=[4],
                    model=model,
                    coupling=1.0,
                    field=0.3,
                    anisotropy=anisotropy,
                )
                terms = [PauliTerm(**term) for term in build_spin_hamiltonian(spec)["terms"]]
                result = run_dmrg(np, DMRGPayload(
                    n_qubits=4,
                    terms=terms,
                    bond_dim=4,
                    sweeps=5,
                    tolerance=1e-5,
                    residual_tolerance=1e-4,
                    lanczos_maxiter=16,
                ))
                exact = exact_ground_state(np, GroundStatePayload(n_qubits=4, terms=terms))
                self.assertLess(abs(result["ground_energy"] - exact["ground_energy"]), 5e-5)
                self.assertAlmostEqual(result["norm2"], 1.0, places=5)
                self.assertTrue(result["converged"])

    def test_dmrg_skips_automatic_exact_check_above_bounded_size(self):
        result = run_dmrg(np, DMRGPayload(
            n_qubits=9,
            terms=[PauliTerm(paulis={index: "Z"}, coefficient=0.2) for index in range(9)],
            bond_dim=2,
            sweeps=2,
            lanczos_maxiter=8,
        ))
        self.assertFalse(result["exact_cross_check"]["performed"])
        self.assertIn("limited to 8 qubits", result["exact_cross_check"]["reason"])

    def test_dmrg_does_not_call_energy_stability_convergence_with_large_variance(self):
        terms = [
            *[PauliTerm(paulis={index: "Z"}, coefficient=0.2) for index in range(4)],
            *[PauliTerm(paulis={index: "X", index + 1: "X"}, coefficient=0.7) for index in range(3)],
        ]
        result = run_dmrg(np, DMRGPayload(
            n_qubits=4,
            terms=terms,
            bond_dim=2,
            sweeps=4,
            lanczos_maxiter=16,
            tolerance=1e-7,
            variance_tolerance=1e-7,
        ))
        self.assertFalse(result["converged"])
        self.assertGreater(result["energy_variance"], 1e-7)
        self.assertIn("energy variance did not reach the requested tolerance", result["warnings"])
        self.assertEqual(result["research_result"]["status"], "needs_review")
        self.assertIn("energy_variance", result["history"][-1])

    def test_dmrg_checkpoint_resume_matches_a_fresh_bounded_run(self):
        terms = [
            *[PauliTerm(paulis={index: "Z"}, coefficient=0.2) for index in range(4)],
            *[PauliTerm(paulis={index: "X", index + 1: "X"}, coefficient=0.7) for index in range(3)],
        ]
        common = {
            "n_qubits": 4,
            "terms": terms,
            "bond_dim": 2,
            "tolerance": 1e-12,
            "residual_tolerance": 1e-5,
            "variance_tolerance": 1e-12,
            "lanczos_maxiter": 8,
        }
        with TemporaryDirectory() as directory:
            checkpoint_path = f"{directory}/dmrg-state.npz"
            partial = run_dmrg(np, DMRGPayload(**common, sweeps=2, checkpoint_path=checkpoint_path))
            self.assertEqual(partial["checkpoint"]["schema"], "quantum-circuit/checkpoint-v1")
            self.assertEqual(partial["checkpoint"]["step"], partial["sweeps_completed"])

            resumed = run_dmrg(np, DMRGPayload(
                **common,
                sweeps=4,
                resume_from=checkpoint_path,
                checkpoint_path=checkpoint_path,
            ))
            fresh = run_dmrg(np, DMRGPayload(**common, sweeps=4))
            self.assertEqual(resumed["sweeps_completed"], 4)
            self.assertAlmostEqual(resumed["ground_energy"], fresh["ground_energy"], places=6)
            self.assertEqual(
                resumed["research_result"]["checkpoint"]["request_sha256"],
                partial["research_result"]["checkpoint"]["request_sha256"],
            )

    def test_dmrg_cancellation_stops_before_returning_a_partial_success(self):
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 2

        with self.assertRaisesRegex(RuntimeError, "job canceled"):
            run_dmrg(np, DMRGPayload(
                n_qubits=4,
                terms=[PauliTerm(paulis={index: "Z"}, coefficient=0.5) for index in range(4)],
                bond_dim=2,
                sweeps=4,
            ), cancel_cb=cancel)
        self.assertGreaterEqual(calls, 2)

    def test_native_peps_evolves_a_two_dimensional_edge(self):
        payload = PEPSPayload(
            n_qubits=4,
            lattice={"dimensions": [2, 2], "boundary": "open"},
            terms=[PauliTerm(paulis={0: "X", 1: "X"}, coefficient=1.0)],
            observables=[PauliTerm(paulis={0: "Z"}, coefficient=1.0, label="Z0")],
            bond_dim=2,
            dt=0.2,
            steps=1,
            order=1,
        )
        result = run_peps(np, payload)
        self.assertEqual(result["backend"], "tensor-network-peps-simple-update")
        self.assertTrue(result["native_geometry"])
        self.assertAlmostEqual(result["expectations"][1]["values"][0], math.cos(0.4), places=5)
        self.assertAlmostEqual(result["norm2"], 1.0, places=5)
        self.assertEqual(result["research_result"]["representation"], "peps")
        self.assertEqual(result["research_result"]["status"], "needs_review")

    def test_native_peps_opt_einsum_matches_bounded_fallback_and_supports_3d(self):
        base = PEPSPayload(
            n_qubits=4,
            lattice={"dimensions": [2, 2], "boundary": "open"},
            terms=[PauliTerm(paulis={0: "X", 1: "X"}, coefficient=0.3)],
            bond_dim=2,
            dt=0.2,
            steps=1,
            order=1,
        )
        optimized = run_peps(np, base)
        enumerated = run_peps(np, base.model_copy(update={"contraction_method": "enumeration"}))
        self.assertAlmostEqual(optimized["norm2"], enumerated["norm2"], places=5)
        self.assertAlmostEqual(optimized["energies"][-1], enumerated["energies"][-1], places=5)
        three_dimensional = run_peps(np, PEPSPayload(
            n_qubits=8,
            lattice={"dimensions": [2, 2, 2], "boundary": "open"},
            terms=[PauliTerm(paulis={0: "X", 1: "X"}, coefficient=0.2)],
            bond_dim=2,
            steps=1,
        ))
        self.assertEqual(three_dimensional["dimensions"], [2, 2, 2])
        self.assertAlmostEqual(three_dimensional["norm2"], 1.0, places=5)

    def test_native_peps_double_layer_runs_past_statevector_bound(self):
        """A 5x5 PEPS must not regress to an exponential statevector path."""
        result = run_peps(np, PEPSPayload(
            n_qubits=25,
            lattice={"dimensions": [5, 5], "boundary": "open"},
            terms=[PauliTerm(paulis={0: "X", 1: "X"}, coefficient=0.2)],
            observables=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            bond_dim=2,
            dt=0.05,
            steps=1,
        ))
        self.assertEqual(result["n_qubits"], 25)
        self.assertFalse(result["contraction_method"] == "enumeration")
        self.assertAlmostEqual(result["norm2"], 1.0, places=5)
        self.assertEqual(len(result["expectations"]), 2)
        self.assertFalse(result["resource_estimate"]["materializes_statevector"])

    def test_boundary_mps_matches_exact_double_layer_at_sufficient_bond(self):
        boundary_payload = PEPSPayload(
            n_qubits=4,
            lattice={"dimensions": [2, 2]},
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            bond_dim=2,
            contraction_method="boundary-mps",
            boundary_bond_dim=16,
        )
        exact_payload = boundary_payload.model_copy(update={"contraction_method": "opt_einsum"})
        boundary = PEPSRuntime(np, boundary_payload)
        rng = np.random.default_rng(23)
        tensors = [
            rng.normal(size=(2, *([2] * len(edge_ids))))
            + 1j * rng.normal(size=(2, *([2] * len(edge_ids))))
            for edge_ids in boundary.site_edges
        ]
        boundary.tensors = tensors
        exact = PEPSRuntime(np, exact_payload)
        exact.tensors = [tensor.copy() for tensor in tensors]
        self.assertAlmostEqual(
            boundary._contract_double_layer({0: "Z", 3: "X"}),
            exact._contract_double_layer({0: "Z", 3: "X"}),
            places=5,
        )
        self.assertGreater(boundary.boundary_summary["boundary_bond_dim_used"], 1)

    def test_boundary_mps_3x3_matches_exact_reference_and_reports_rows(self):
        boundary_payload = PEPSPayload(
            n_qubits=9,
            lattice={"dimensions": [3, 3]},
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            bond_dim=2,
            contraction_method="boundary-mps",
            boundary_bond_dim=16,
        )
        exact_payload = boundary_payload.model_copy(update={"contraction_method": "opt_einsum"})
        enumerated_payload = boundary_payload.model_copy(update={"contraction_method": "enumeration"})
        boundary = PEPSRuntime(np, boundary_payload)
        exact = PEPSRuntime(np, exact_payload)
        enumerated = PEPSRuntime(np, enumerated_payload)
        rng = np.random.default_rng(107)
        tensors = [
            (0.2 * (rng.normal(size=(2, *([2] * len(edge_ids)))
                    ) + 1j * rng.normal(size=(2, *([2] * len(edge_ids)))))).astype(np.complex64)
            for edge_ids in boundary.site_edges
        ]
        boundary.tensors = tensors
        exact.tensors = [tensor.copy() for tensor in tensors]
        enumerated.tensors = [tensor.copy() for tensor in tensors]
        actual = boundary._contract_double_layer({0: "Z", 4: "X", 8: "Z"})
        expected = exact._contract_double_layer({0: "Z", 4: "X", 8: "Z"})
        independent_reference = enumerated._contract_double_layer({0: "Z", 4: "X", 8: "Z"})
        self.assertLess(abs(actual - expected), 1e-3)
        self.assertLess(abs(actual - independent_reference), 1e-3)
        self.assertEqual(len(boundary.boundary_summary["rows"]), 3)
        self.assertEqual(boundary.boundary_summary["rows_completed"], 3)
        self.assertEqual(boundary.boundary_summary["discarded_weight"], 0.0)
        self.assertTrue(boundary.boundary_summary["converged"])

    def test_boundary_mps_chi_study_reports_truncation_and_convergence(self):
        low_payload = PEPSPayload(
            n_qubits=9,
            lattice={"dimensions": [3, 3]},
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            bond_dim=2,
            contraction_method="boundary-mps",
            boundary_bond_dim=1,
        )
        high_payload = low_payload.model_copy(update={"boundary_bond_dim": 16})
        exact_payload = low_payload.model_copy(update={"contraction_method": "opt_einsum"})
        rng = np.random.default_rng(109)
        reference_runtime = PEPSRuntime(np, exact_payload)
        tensors = [
            (0.2 * (rng.normal(size=(2, *([2] * len(edge_ids)))
                    ) + 1j * rng.normal(size=(2, *([2] * len(edge_ids)))))).astype(np.complex64)
            for edge_ids in reference_runtime.site_edges
        ]
        reference_runtime.tensors = [tensor.copy() for tensor in tensors]
        reference = reference_runtime._contract_double_layer({0: "Z", 8: "X"})
        low_runtime = PEPSRuntime(np, low_payload)
        high_runtime = PEPSRuntime(np, high_payload)
        low_runtime.tensors = [tensor.copy() for tensor in tensors]
        high_runtime.tensors = [tensor.copy() for tensor in tensors]
        low = low_runtime._contract_double_layer({0: "Z", 8: "X"})
        high = high_runtime._contract_double_layer({0: "Z", 8: "X"})
        self.assertGreater(low_runtime.boundary_summary["discarded_weight"], 0.0)
        self.assertEqual(high_runtime.boundary_summary["discarded_weight"], 0.0)
        self.assertLess(abs(high - reference), abs(low - reference))
        self.assertEqual([row["row"] for row in high_runtime.boundary_summary["rows"]], [1, 2, 3])

    def test_peps_reports_physical_and_boundary_truncation_separately(self):
        spec = LatticeHamiltonianPayload(
            dimensions=[3, 3],
            model="heisenberg",
            coupling=1.0,
            field=0.3,
        )
        payload = PEPSPayload(
            n_qubits=9,
            lattice={"dimensions": [3, 3], "boundary": "open"},
            terms=[PauliTerm(**term) for term in build_spin_hamiltonian(spec)["terms"]],
            observables=[PauliTerm(paulis={0: "Z"}, coefficient=1.0, label="Z0")],
            bond_dim=2,
            boundary_bond_dim=1,
            contraction_method="boundary-mps",
            dt=0.05,
            steps=1,
        )
        result = run_peps(np, payload)
        physical = float(result["discarded_weight"])
        boundary = float(result["boundary_diagnostics"]["discarded_weight"])
        total = float(result["research_result"]["truncation"]["discarded_weight"])
        self.assertGreater(physical, 0.0)
        self.assertGreater(boundary, 0.0)
        self.assertAlmostEqual(total, physical + boundary, places=10)
        self.assertEqual(result["research_result"]["truncation"]["max_bond_dim"], 2)
        self.assertEqual(result["research_result"]["truncation"]["max_environment_dim"], 1)
        self.assertEqual(len(result["expectations"][-1]["values"]), 1)

    def test_boundary_mps_checkpoint_resume_restores_environment(self):
        payload = PEPSPayload(
            n_qubits=9,
            lattice={"dimensions": [3, 3]},
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            bond_dim=2,
            contraction_method="boundary-mps",
            boundary_bond_dim=4,
        )
        source = PEPSRuntime(np, payload)
        rng = np.random.default_rng(113)
        tensors = [
            (0.2 * (rng.normal(size=(2, *([2] * len(edge_ids)))
                    ) + 1j * rng.normal(size=(2, *([2] * len(edge_ids)))))).astype(np.complex64)
            for edge_ids in source.site_edges
        ]
        source.tensors = [tensor.copy() for tensor in tensors]
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 2

        with TemporaryDirectory() as directory:
            checkpoint_path = f"{directory}/boundary-state.npz"
            with self.assertRaisesRegex(RuntimeError, "job canceled"):
                contract_boundary_mps(
                    source,
                    {0: "Z", 8: "X"},
                    max_bond_dim=4,
                    checkpoint_path=checkpoint_path,
                    cancel_cb=cancel,
                )
            resumed_payload = payload.model_copy(update={
                "boundary_resume_from": checkpoint_path,
                "boundary_checkpoint_path": checkpoint_path,
            })
            resumed = PEPSRuntime(np, resumed_payload)
            resumed.tensors = [tensor.copy() for tensor in tensors]
            resumed_value = resumed._contract_double_layer({0: "Z", 8: "X"})
            full = PEPSRuntime(np, payload)
            full.tensors = [tensor.copy() for tensor in tensors]
            full_value = full._contract_double_layer({0: "Z", 8: "X"})
            self.assertAlmostEqual(resumed_value, full_value, places=4)
            self.assertEqual(resumed.boundary_summary["start_row"], 1)
            self.assertEqual(resumed.boundary_summary["rows_completed"], 3)
            self.assertEqual(resumed.boundary_summary["checkpoint"]["step"], 3)

    def test_boundary_mps_reports_environment_truncation(self):
        payload = PEPSPayload(
            n_qubits=4,
            lattice={"dimensions": [2, 2]},
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            bond_dim=2,
            contraction_method="boundary-mps",
            boundary_bond_dim=1,
        )
        runtime = PEPSRuntime(np, payload)
        rng = np.random.default_rng(31)
        runtime.tensors = [
            rng.normal(size=(2, *([2] * len(edge_ids))))
            + 1j * rng.normal(size=(2, *([2] * len(edge_ids))))
            for edge_ids in runtime.site_edges
        ]
        value = runtime._contract_double_layer({})
        self.assertTrue(np.isfinite(value))
        self.assertGreater(runtime.boundary_summary["discarded_weight"], 0.0)

    def test_dmrg_and_peps_preflight_report_bounded_local_work(self):
        dmrg_payload = DMRGPayload(
            n_qubits=4,
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            bond_dim=4,
            max_local_dim=64,
        )
        dmrg_report = estimate_dmrg(dmrg_payload, gpu_free_mb=1024)
        self.assertTrue(dmrg_report["feasible"])
        peps_payload = PEPSPayload(
            n_qubits=4,
            lattice={"dimensions": [2, 2]},
            terms=[PauliTerm(paulis={0: "Z"}, coefficient=1.0)],
            bond_dim=2,
        )
        peps_report = estimate_peps(peps_payload, gpu_free_mb=1024)
        self.assertTrue(peps_report["feasible"])
        self.assertEqual(peps_report["virtual_bond_states"], 16)
        self.assertEqual(peps_report["double_layer_bond_dim"], 4)
        self.assertFalse(peps_report["materializes_statevector"])
        enumeration_report = estimate_peps(PEPSPayload(
            n_qubits=25,
            lattice={"dimensions": [5, 5]},
            terms=peps_payload.terms,
            bond_dim=2,
            contraction_method="enumeration",
        ))
        self.assertFalse(enumeration_report["feasible"])
        self.assertTrue(any("16-site" in warning for warning in enumeration_report["blocking_warnings"]))
        boundary_report = estimate_peps(PEPSPayload(
            n_qubits=8,
            lattice={"dimensions": [2, 2, 2]},
            terms=peps_payload.terms,
            bond_dim=2,
            contraction_method="boundary-mps",
        ))
        self.assertFalse(boundary_report["feasible"])
        self.assertTrue(any("unsupported" in warning for warning in boundary_report["blocking_warnings"]))

    def test_mps_expectation_returns_bell_correlations(self):
        runtime = MPSRuntime(np, TNPayload(
            n_qubits=2,
            gates=[TNGate(name="h", target=0), TNGate(name="cx", control=0, target=1)],
            bond_dim=4,
        ))
        terms = [
            PauliTerm(paulis={0: "X", 1: "X"}, coefficient=1.0),
            PauliTerm(paulis={0: "Z", 1: "Z"}, coefficient=1.0),
        ]
        for value in runtime.expectation(terms):
            self.assertAlmostEqual(value, 1.0, places=5)
        self.assertAlmostEqual(runtime.norm2(), 1.0, places=6)

    def test_tebd_keeps_product_state_norm(self):
        payload = TEBDPayload(
            n_qubits=4,
            terms=[PauliTerm(paulis={0: "Z", 1: "Z"}, coefficient=1.0)],
            observables=[PauliTerm(paulis={0: "Z"}, coefficient=1.0, label="Z0")],
            dt=0.05,
            steps=2,
            bond_dim=4,
        )
        result = run_tebd(np, payload)
        self.assertEqual(result["backend"], "tensor-network-mps-tebd")
        self.assertEqual(len(result["expectations"]), 3)
        self.assertEqual(len(result["energies"]), 3)
        self.assertAlmostEqual(result["expectations"][0]["energy"], result["energies"][0], places=6)
        self.assertAlmostEqual(result["norm2"], 1.0, places=5)
        self.assertIn("norm_drift", result)
        self.assertIn("bond_growth", result)
        self.assertIn("truncation_diagnostics", result)
        self.assertEqual(len(result["norm_history"]), 3)
        self.assertEqual(len(result["bond_dim_history"]), 3)
        self.assertAlmostEqual(result["norm_drift"], max(abs(value - 1.0) for value in result["norm_history"]), places=8)
        self.assertEqual(result["expectations"][-1]["bond_growth"], result["bond_growth"])

    def test_tebd_cancellation_stops_before_returning_a_partial_success(self):
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 2

        with self.assertRaisesRegex(RuntimeError, "job canceled"):
            run_tebd(np, TEBDPayload(
                n_qubits=4,
                terms=[PauliTerm(paulis={0: "Z", 1: "Z"}, coefficient=0.5)],
                steps=4,
                bond_dim=2,
            ), cancel_cb=cancel)
        self.assertGreaterEqual(calls, 2)

    def test_tebd_preflight_prices_nonlocal_swap_work(self):
        local = TEBDPayload(
            n_qubits=8,
            terms=[PauliTerm(paulis={0: "Z", 1: "Z"}, coefficient=1.0)],
            steps=2,
            bond_dim=4,
        )
        long_range = local.model_copy(update={"terms": [PauliTerm(paulis={0: "Z", 7: "Z"}, coefficient=1.0)]})
        local_report = estimate_tebd(local, gpu_free_mb=1024)
        long_range_report = estimate_tebd(long_range, gpu_free_mb=1024)
        self.assertTrue(local_report["feasible"])
        self.assertGreater(long_range_report["interaction_cost"], local_report["interaction_cost"])

    def test_tebd_two_site_pauli_evolution_matches_known_state(self):
        payload = TEBDPayload(
            n_qubits=2,
            terms=[PauliTerm(paulis={0: "X", 1: "Y"}, coefficient=1.0)],
            observables=[PauliTerm(paulis={0: "Z"}, coefficient=1.0, label="Z0")],
            dt=0.2,
            steps=1,
            order=1,
            bond_dim=4,
        )
        result = run_tebd(np, payload)
        self.assertAlmostEqual(result["expectations"][1]["values"][0], math.cos(0.4), places=4)

    def test_tebd_three_site_pauli_string_evolution_matches_known_state(self):
        payload = TEBDPayload(
            n_qubits=3,
            terms=[PauliTerm(paulis={0: "X", 1: "Y", 2: "Z"}, coefficient=1.0)],
            observables=[PauliTerm(paulis={0: "Z"}, coefficient=1.0, label="Z0")],
            dt=0.2,
            steps=1,
            order=1,
            bond_dim=4,
        )
        result = run_tebd(np, payload)
        self.assertAlmostEqual(result["expectations"][1]["values"][0], math.cos(0.4), places=4)
        self.assertAlmostEqual(result["norm2"], 1.0, places=5)
        self.assertEqual(result["max_term_locality"], 3)
        self.assertEqual(result["parity_string_terms"], 1)

    def test_expectation_payload_checks_sparse_term_indices(self):
        with self.assertRaises(ValueError):
            ExpectationPayload(
                n_qubits=2,
                terms=[PauliTerm(paulis={2: "Z"}, coefficient=1.0)],
            )


if __name__ == "__main__":
    unittest.main()
