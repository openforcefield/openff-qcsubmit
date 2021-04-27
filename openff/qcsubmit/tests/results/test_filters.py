import logging

import pytest
from openff.toolkit.topology import Molecule
from pydantic import ValidationError
from qcelemental.models import DriverEnum
from qcportal.models import ObjectId, ResultRecord
from qcportal.models.records import RecordStatusEnum

from openff.qcsubmit.results import BasicResult
from openff.qcsubmit.results.filters import (
    CMILESResultFilter,
    ConnectivityFilter,
    HydrogenBondFilter,
    RecordStatusFilter,
    ResultFilter,
    ResultRecordFilter,
    SMARTSFilter,
    SMILESFilter,
)


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
            return result.record_id == "1"

    filtered_collection = DummyFilter().apply(basic_result_collection)

    assert filtered_collection.n_results == 2

    for port in [442, 443]:

        address = f"http://localhost:{port}"

        assert address in filtered_collection.entries
        assert len(filtered_collection.entries[address]) == 1
        assert filtered_collection.entries[address][0].record_id == "1"


def test_apply_record_filter(basic_result_collection):
    class DummyFilter(ResultRecordFilter):
        def _filter_function(self, result, record, molecule) -> bool:
            return record.client.address == "http://localhost:442"

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


@pytest.mark.parametrize(
    "result_filter, expected_ids",
    [
        (SMILESFilter(smiles_to_include=["CCO"]), {"http://localhost:442": {"2", "3"}}),
        (
            SMILESFilter(smiles_to_exclude=["CCO"]),
            {
                "http://localhost:442": {"1", "4"},
                "http://localhost:443": {"1", "2", "3", "4"},
            },
        ),
        (
            SMARTSFilter(smarts_to_include=["[#6]-[#8H1]"]),
            {"http://localhost:442": {"1", "2", "3", "4"}},
        ),
        (
            SMARTSFilter(smarts_to_exclude=["[#6]-[#8]"]),
            {"http://localhost:443": {"1", "2", "3", "4"}},
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


@pytest.mark.parametrize(
    "result_filter, expected_ids",
    [
        (
            HydrogenBondFilter(method="baker-hubbard"),
            {"http://localhost:443": {"1"}},
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
            {"http://localhost:442": {"1", "2", "3", "4"}},
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
        (HydrogenBondFilter(method="baker-hubbard"), {"http://localhost:443": {"1"}}),
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
        record_id=ObjectId("1"),
        cmiles="[Cl:1][Cl:2]",
        inchi_key="KZBUYRJDOAKODT-UHFFFAOYSA-N",
    )
    record = ResultRecord(
        id=ObjectId("1"),
        program="psi4",
        driver=DriverEnum.gradient,
        method="scf",
        basis="sto-3g",
        molecule=ObjectId("1"),
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

    record = ResultRecord(
        id=ObjectId("1"),
        program="psi4",
        driver=DriverEnum.gradient,
        method="scf",
        basis="sto-3g",
        molecule=ObjectId("1"),
        status=RecordStatusEnum.complete,
    )

    status_filter = RecordStatusFilter(status=RecordStatusEnum.complete)
    assert status_filter._filter_function(None, record, None) is True

    status_filter = RecordStatusFilter(status=RecordStatusEnum.incomplete)
    assert status_filter._filter_function(None, record, None) is False
