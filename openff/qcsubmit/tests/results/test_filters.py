import logging

import numpy
import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit
from pydantic import ValidationError
from qcelemental.models import DriverEnum
#from qcportal.records import RecordStatusEnum, SinglepointRecord
from . import RecordStatusEnum, SinglepointRecord

from openff.qcsubmit.results import (
    BasicResult,
    OptimizationResult,
    OptimizationResultCollection,
)
from openff.qcsubmit.results.filters import (
    ChargeFilter,
    CMILESResultFilter,
    ConformerRMSDFilter,
    ConnectivityFilter,
    ElementFilter,
    HydrogenBondFilter,
    LowestEnergyFilter,
    MinimumConformersFilter,
    RecordStatusFilter,
    ResultFilter,
    SinglepointRecordFilter,
    SMARTSFilter,
    SMILESFilter,
    UnperceivableStereoFilter,
)
from openff.qcsubmit.tests.results import mock_optimization_result_collection


def test_apply_filter(basic_result_collection, caplog):
    class DummyFilter(ResultFilter):
        def _apply(self, result_collection):

            result_collection.entries = {
                "http://localhost:442": result_collection.entries[
                    "http://localhost:442"
                ]
            }

            return result_collection

    with caplog.at_level(logging.INFO):
        filtered_collection = DummyFilter().apply(basic_result_collection)

    assert filtered_collection.n_results == 4
    assert "4 results were removed" in caplog.text

    assert "applied-filters" in filtered_collection.provenance
    assert "DummyFilter-0" in filtered_collection.provenance["applied-filters"]


def test_apply_cmiles_filter(basic_result_collection):

    class DummyFilter(CMILESResultFilter):
        def _filter_function(self, result) -> bool:
            #1/0
            return result.record_id == 1

    filtered_collection = DummyFilter().apply(basic_result_collection)

    assert filtered_collection.n_results == 2

    for port in [442, 443]:

        address = f"http://localhost:{port}"

        assert address in filtered_collection.entries
        assert len(filtered_collection.entries[address]) == 1
        assert filtered_collection.entries[address][0].record_id == 1


def test_apply_record_filter(basic_result_collection):
    class DummyFilter(SinglepointRecordFilter):
        def _filter_function(self, result, record, molecule) -> bool:
            return record._client.address == "http://localhost:442"

    filtered_collection = DummyFilter().apply(basic_result_collection)

    assert filtered_collection.n_results == 4

    assert "http://localhost:442" in filtered_collection.entries
    assert "http://localhost:443" not in filtered_collection.entries


def test_smiles_filter_mutual_inputs():

    with pytest.raises(ValidationError, match="exactly one of `smiles_to_include`"):
        SMILESFilter(smiles_to_include=["C"], smiles_to_exclude=["CC"])


def test_smarts_filter_mutual_inputs():

    with pytest.raises(ValidationError, match="exactly one of `smarts_to_include`"):
        SMARTSFilter(smarts_to_include=["C"], smarts_to_exclude=["CC"])


def test_charge_filter_mutual_inputs():

    with pytest.raises(ValidationError, match="exactly one of `charges_to_include`"):
        ChargeFilter(charges_to_include=[0], charges_to_exclude=[1, 2])


@pytest.mark.parametrize(
    "result_filter, expected_ids",
    [
        (SMILESFilter(smiles_to_include=["CCO"]), {"http://localhost:442": {2, 3}}),
        (
            SMILESFilter(smiles_to_exclude=["CCO"]),
            {
                "http://localhost:442": {1, 4},
                "http://localhost:443": {1, 2, 3, 4},
            },
        ),
        (
            SMARTSFilter(smarts_to_include=["[#6]-[#8H1]"]),
            {"http://localhost:442": {1, 2, 3, 4}},
        ),
        (
            SMARTSFilter(smarts_to_exclude=["[#6]-[#8]"]),
            {"http://localhost:443": {1, 2, 3, 4}},
        ),
    ],
)
def test_molecule_filter_apply(result_filter, expected_ids, basic_result_collection):

    filtered_collection = result_filter.apply(basic_result_collection)

    assert {*expected_ids} == {*filtered_collection.entries}

    for address, entry_ids in expected_ids.items():

        assert entry_ids == {
            entry.record_id for entry in filtered_collection.entries[address]
        }


def test_molecule_filter_tautomers(tautomer_basic_result_collection):
    """Filter for only one tautomer to ensure we are using the fixed hydrogen inchikey."""

    result_filter = SMILESFilter(smiles_to_include=["C1=NC(=O)NN=C1"])

    filtered_collection = result_filter.apply(tautomer_basic_result_collection)

    assert filtered_collection.n_molecules == 1
    assert len(filtered_collection.entries["http://localhost:442"]) == 1
    assert filtered_collection.entries["http://localhost:442"][0].record_id == 2


@pytest.mark.parametrize(
    "result_filter, expected_ids",
    [
        (
            HydrogenBondFilter(method="baker-hubbard"),
            {"http://localhost:443": {1}},
        ),
    ],
)
def test_basic_record_filter_apply(
    result_filter, expected_ids, h_bond_basic_result_collection
):

    filtered_collection = result_filter.apply(h_bond_basic_result_collection)

    assert {*expected_ids} == {*filtered_collection.entries}

    for address, entry_ids in expected_ids.items():

        assert entry_ids == {
            entry.record_id for entry in filtered_collection.entries[address]
        }


@pytest.mark.parametrize(
    "result_filter, expected_ids",
    [
        (
            HydrogenBondFilter(method="baker-hubbard"),
            {"http://localhost:442": {1, 2, 3, 4}},
        ),
    ],
)
def test_optimization_record_filter_apply(
    result_filter, expected_ids, optimization_result_collection
):

    filtered_collection = result_filter.apply(optimization_result_collection)

    assert {*expected_ids} == {*filtered_collection.entries}

    for address, entry_ids in expected_ids.items():

        assert entry_ids == {
            entry.record_id for entry in filtered_collection.entries[address]
        }


@pytest.mark.parametrize(
    "result_filter, expected_ids",
    [
        (HydrogenBondFilter(method="baker-hubbard"), {"http://localhost:443": {1}}),
    ],
)
def test_torsion_drive_record_filter_apply(
    result_filter, expected_ids, torsion_drive_result_collection
):

    filtered_collection = result_filter.apply(torsion_drive_result_collection)

    assert {*expected_ids} == {*filtered_collection.entries}

    for address, entry_ids in expected_ids.items():

        assert entry_ids == {
            entry.record_id for entry in filtered_collection.entries[address]
        }


def test_connectivity_filter():

    result = BasicResult(
        record_id=1,
        cmiles="[Cl:1][Cl:2]",
        inchi_key="KZBUYRJDOAKODT-UHFFFAOYSA-N",
    )
    record = SinglepointRecord(
        id=1,
        program="psi4",
        driver=DriverEnum.gradient,
        method="scf",
        basis="sto-3g",
        molecule=1,
        status=RecordStatusEnum.complete,
    )

    connectivity_filter = ConnectivityFilter()

    molecule: Molecule = Molecule.from_smiles("[Cl:1][Cl:2]")
    molecule.generate_conformers(n_conformers=1)

    assert connectivity_filter._filter_function(result, record, molecule)

    molecule.conformers[0] *= 10.0

    assert not connectivity_filter._filter_function(result, record, molecule)

    connectivity_filter.tolerance = 12.01  # default * 10.0 + 0.01
    assert connectivity_filter._filter_function(result, record, molecule)


def test_record_status_filter():

    record = SinglepointRecord(
        id=1,
        program="psi4",
        driver=DriverEnum.gradient,
        method="scf",
        basis="sto-3g",
        molecule=1,
        status=RecordStatusEnum.complete,
    )

    status_filter = RecordStatusFilter(status=RecordStatusEnum.complete)
    assert status_filter._filter_function(None, record, None) is True

    status_filter = RecordStatusFilter(status=RecordStatusEnum.incomplete)
    assert status_filter._filter_function(None, record, None) is False


def test_charge_filter():

    record = BasicResult(record_id=1, cmiles="[N+:1](=[O:2])([O-:3])[O-:4]", inchi_key="NHNBFGGVMKEFGY-UHFFFAOYSA-N")
    charge_filter = ChargeFilter(charges_to_include=[-1, 0])

    assert charge_filter._filter_function(entry=record) is True

    charge_filter = ChargeFilter(charges_to_exclude=[-1])

    assert charge_filter._filter_function(entry=record) is False


def test_element_filter(basic_result_collection):

    # use mixed ints and str
    element_filter = ElementFilter(allowed_elements=[1, 6, "O"])

    result = element_filter.apply(result_collection=basic_result_collection)
    # no molecules are filtered
    assert result.n_results == 8

    # no hydrogen should filter everything
    element_filter.allowed_elements = [6, 8]

    result = element_filter.apply(result_collection=basic_result_collection)
    assert result.n_results == 0
    assert result.n_molecules == 0


def test_lowest_energy_filter(optimization_result_collection_duplicates):

    energy_filter = LowestEnergyFilter()

    # should have 2 results
    assert optimization_result_collection_duplicates.n_results == 2
    result = energy_filter.apply(result_collection=optimization_result_collection_duplicates)

    # make sure we only have one result
    assert result.n_molecules == 1
    assert result.n_results == 1


@pytest.mark.parametrize("min_conformers, n_expected_results", [(1, 2), (3, 0)])
def test_min_conformers_filter(
    optimization_result_collection_duplicates, min_conformers, n_expected_results
):

    min_conformers_filter = MinimumConformersFilter(min_conformers=min_conformers)

    assert optimization_result_collection_duplicates.n_results == 2
    assert optimization_result_collection_duplicates.n_molecules == 1

    result = min_conformers_filter.apply(
        result_collection=optimization_result_collection_duplicates
    )

    assert result.n_results == n_expected_results


@pytest.mark.parametrize(
    ("max_conformers, rmsd_tolerance, heavy_atoms_only, expected_record_ids"),
    [
        # max_conformers
        (1, 0.1, False, {"1"}),
        (2, 0.1, False, {"1", "3"}),
        (3, 0.1, False, {"1", "2", "3"}),
        # rmsd_tolerance
        (3, 0.75, False, {"1", "3"}),
        # heavy_atoms_only
        (1, 0.1, True, {"1"}),
    ],
)
def test_rmsd_conformer_filter(
    max_conformers, rmsd_tolerance, heavy_atoms_only, expected_record_ids, monkeypatch
):

    molecules = []

    for conformer in [
        numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * unit.angstrom,
        numpy.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) * unit.angstrom,
        numpy.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]) * unit.angstrom,
    ]:

        molecule = Molecule.from_smiles(smiles="Cl[H]")
        molecule._conformers = [conformer]

        molecules.append(molecule)

    result_collection = mock_optimization_result_collection(
        {"http://localhost:442": molecules}, monkeypatch
    )

    filtered_collection = result_collection.filter(
        ConformerRMSDFilter(
            max_conformers=max_conformers,
            rmsd_tolerance=rmsd_tolerance,
            heavy_atoms_only=heavy_atoms_only,
            check_automorphs=False,
        )
    )

    found_entry_ids = {
        entry.record_id
        for entries in filtered_collection.entries.values()
        for entry in entries
    }

    assert found_entry_ids == expected_record_ids


def test_rmsd_conformer_filter_canonical_order(monkeypatch):

    molecule_a = Molecule.from_mapped_smiles("[Cl:1][H:2]")
    molecule_a._conformers = [
        numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * unit.angstrom
    ]
    molecule_b = Molecule.from_mapped_smiles("[Cl:2][H:1]")
    molecule_b._conformers = [
        numpy.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) * unit.angstrom
    ]

    result_collection = mock_optimization_result_collection(
        {"http://localhost:442": [molecule_a, molecule_b]}, monkeypatch
    )

    filtered_collection = result_collection.filter(ConformerRMSDFilter())

    assert filtered_collection.n_molecules == 1
    assert filtered_collection.n_results == 1


@pytest.mark.parametrize(
    "rmsd_function_name",
    [
        "_compute_rmsd_matrix_rd",
        "_compute_rmsd_matrix_oe",
        "_compute_rmsd_matrix",
    ],
)
def test_rmsd_conformer_filter_rmsd_matrix(rmsd_function_name):

    molecule = Molecule.from_mapped_smiles("[O:1]=[C:2]=[O:3]")
    molecule._conformers = [
        numpy.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        * unit.angstrom,
        numpy.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        * unit.angstrom,
    ]

    rmsd_filter = ConformerRMSDFilter(check_automorphs=True, heavy_atoms_only=False)
    rmsd_function = getattr(rmsd_filter, rmsd_function_name)

    actual_rmsd_matrix = rmsd_function(molecule)
    expected_rmsd_matrix = numpy.array(
        [[0.0, numpy.sqrt(2.0 / 3)], [numpy.sqrt(2.0 / 3), 0.0]]
    )

    assert numpy.allclose(actual_rmsd_matrix, expected_rmsd_matrix)


@pytest.mark.parametrize(
    "rmsd_function_name",
    [
        "_compute_rmsd_matrix_rd",
        "_compute_rmsd_matrix_oe",
        "_compute_rmsd_matrix",
    ],
)
@pytest.mark.parametrize(
    "heavy_atoms_only, expected_rmsd_matrix",
    [
        (False, numpy.array([[0.0, 0.5], [0.5, 0.0]])),
        (True, numpy.array([[0.0, 0.0], [0.0, 0.0]])),
    ],
)
def test_rmsd_conformer_filter_rmsd_matrix_heavy_only(
    rmsd_function_name, heavy_atoms_only, expected_rmsd_matrix
):

    molecule = Molecule.from_smiles("Cl[H]")
    molecule._conformers = [
        numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * unit.angstrom,
        numpy.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) * unit.angstrom,
    ]

    rmsd_filter = ConformerRMSDFilter(
        check_automorphs=True, heavy_atoms_only=heavy_atoms_only
    )
    rmsd_function = getattr(rmsd_filter, rmsd_function_name)

    actual_rmsd_matrix = rmsd_function(molecule)

    assert numpy.allclose(actual_rmsd_matrix, expected_rmsd_matrix)


@pytest.mark.parametrize(
    "rmsd_function_name",
    [
        "_compute_rmsd_matrix_rd",
        "_compute_rmsd_matrix_oe",
        "_compute_rmsd_matrix",
    ],
)
@pytest.mark.parametrize(
    "check_automorphs, expected_rmsd_matrix",
    [
        (False, numpy.array([[0.0, numpy.sqrt(8 / 6)], [numpy.sqrt(8 / 6), 0.0]])),
        (True, numpy.array([[0.0, 0.0], [0.0, 0.0]])),
    ],
)
def test_rmsd_conformer_filter_rmsd_matrix_automorphs(
    rmsd_function_name, check_automorphs, expected_rmsd_matrix
):

    molecule = Molecule.from_mapped_smiles("[Br:3][C:1]([Cl:4])=[C:2]([Cl:6])[Cl:5]")
    molecule._conformers = [
        numpy.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-2.0, 1.0, 0.0],
                [-2.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, -1.0, 0.0],
            ]
        )
        * unit.angstrom,
        numpy.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-2.0, 1.0, 0.0],
                [-2.0, -1.0, 0.0],
                [2.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        * unit.angstrom,
    ]

    rmsd_filter = ConformerRMSDFilter(
        check_automorphs=check_automorphs, heavy_atoms_only=False
    )
    rmsd_function = getattr(rmsd_filter, rmsd_function_name)

    actual_rmsd_matrix = rmsd_function(molecule)

    assert numpy.allclose(actual_rmsd_matrix, expected_rmsd_matrix)


@pytest.mark.parametrize(
    "toolkits, n_expected",
    [
        (["rdkit"], 1),
        (["openeye"], 0),
        (["openeye", "rdkit"], 0),
    ]
)
def test_unperceivable_stereo_filter(toolkits, n_expected, public_client):

    collection = OptimizationResultCollection(
        entries={
            "https://api.qcarchive.molssi.org:443/": [
                OptimizationResult(
                    record_id=19095884,
                    cmiles=(
                        "[H:37][c:1]1[c:3]([c:8]([c:6]([c:9]([c:4]1[H:40])[S:36]"
                        "(=[O:32])(=[O:33])[N:29]2[C:17]([C:21]([C:18]2([H:53])[H:54])"
                        "([F:34])[F:35])([H:51])[H:52])[H:42])[N:30]([H:66])[c:11]3"
                        "[c:5]([c:2]([c:7]4[c:10]([n:25]3)[N@@:27]([C@:19]([C:12]"
                        "(=[O:31])[N:26]4[C:23]([H:60])([H:61])[H:62])([H:55])[C:22]"
                        "([H:57])([H:58])[H:59])[C:20]5([C:13]([C:15]([N:28]([C:16]"
                        "([C:14]5([H:45])[H:46])([H:49])[H:50])[C:24]([H:63])([H:64])"
                        "[H:65])([H:47])[H:48])([H:43])[H:44])[H:56])[H:38])[H:41])"
                        "[H:39]"
                    ),
                    inchi_key="GMRICROFHKBHBU-MRXNPFEDSA-N"
                )
            ]
        }
    )
    assert collection.n_results == 1

    filtered = collection.filter(UnperceivableStereoFilter(toolkits=toolkits))
    assert filtered.n_results == n_expected
