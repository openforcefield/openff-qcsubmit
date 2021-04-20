"""
Test the results packages when collecting from the public qcarchive.
"""

import pytest
from pydantic import ValidationError
from qcportal import FractalClient
from qcportal.models import ObjectId, OptimizationRecord, ResultRecord
from qcportal.models.common_models import DriverEnum, QCSpecification
from qcportal.models.records import RecordStatusEnum

from openff.qcsubmit.exceptions import RecordTypeError
from openff.qcsubmit.results import (
    BasicResult,
    BasicResultCollection,
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.results.results import OptimizationResult, _BaseResultCollection
from openff.qcsubmit.tests import does_not_raise


@pytest.fixture
def public_client():
    """Setup a new connection to the public qcarchive client."""

    return FractalClient()


def test_base_molecule_property():

    record = BasicResult(
        record_id=ObjectId("1"),
        cmiles="[Cl:2][H:1]",
        inchi_key="VEXZGXHMUGYJMC-UHFFFAOYSA-N",
    )
    molecule = record.molecule

    assert molecule.atoms[0].atomic_number == 1
    assert molecule.atoms[1].atomic_number == 17


@pytest.mark.parametrize(
    "entries, expected_raises",
    [
        (
            [
                BasicResult(
                    record_id=ObjectId("1"),
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                )
            ],
            does_not_raise(),
        ),
        (
            [
                BasicResult(
                    record_id=ObjectId("1"),
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                )
            ]
            * 2,
            pytest.raises(ValidationError, match="duplicate entries were found for"),
        ),
    ],
)
def test_base_validate_entries(entries, expected_raises):

    with expected_raises:
        collection = BasicResultCollection(entries={"http://localhost:443": entries})

        assert isinstance(collection.entries, dict)
        assert len(collection.entries["http://localhost:443"]) == len(entries)


def test_base_n_results_property():

    collection = BasicResultCollection(
        entries={
            "http://localhost:442": [
                BasicResult(
                    record_id=ObjectId("1"),
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                )
            ],
            "http://localhost:443": [
                BasicResult(
                    record_id=ObjectId("1"),
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                )
            ],
        }
    )

    assert collection.n_results == 2


def test_base_n_molecules_property():

    collection = BasicResultCollection(
        entries={
            "http://localhost:442": [
                BasicResult(
                    record_id=ObjectId("1"),
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                )
            ],
            "http://localhost:443": [
                BasicResult(
                    record_id=ObjectId("1"),
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                ),
                BasicResult(
                    record_id=ObjectId("2"),
                    cmiles="[Cl:1][Cl:2]",
                    inchi_key="KZBUYRJDOAKODT-UHFFFAOYSA-N",
                ),
            ],
        }
    )

    assert collection.n_molecules == 2


def test_base_query_in_chunks():

    n_queries = 0

    def _query_function(ids):

        nonlocal n_queries
        n_queries += 1

        return [int(i) + 1 for i in ids]

    result = _BaseResultCollection._query_in_chunks(
        _query_function, ["1", "2", "3", "4", "5"], 2
    )

    assert result == [2, 3, 4, 5, 6]
    assert n_queries == 3


def test_base_validate_record_types():

    records = [
        ResultRecord(
            program="psi4",
            driver=DriverEnum.gradient,
            method="scf",
            basis="sto-3g",
            molecule=ObjectId("1"),
            status=RecordStatusEnum.complete,
        ),
        OptimizationRecord(
            program="psi4",
            qc_spec=QCSpecification(
                driver=DriverEnum.gradient, method="scf", basis="sto-3g", program="psi4"
            ),
            initial_molecule=ObjectId("1"),
            status=RecordStatusEnum.complete,
        ),
    ]

    _BaseResultCollection._validate_record_types(records[:1], ResultRecord)
    _BaseResultCollection._validate_record_types(records[1:], OptimizationRecord)

    with pytest.raises(RecordTypeError, match="The collection contained a records"):
        _BaseResultCollection._validate_record_types(records, OptimizationRecord)


@pytest.mark.parametrize(
    "collection_type, dataset, spec, n_molecules, n_results",
    [
        (
            BasicResultCollection,
            "OpenFF BCC Refit Study COH v1.0",
            "resp-2-vacuum",
            94,
            429,
        ),
        (
            OptimizationResultCollection,
            "OpenFF Protein Fragments v1.0",
            "default",
            # The optimization collection is small containing only 576 records however
            # some are incomplete and only 16 unique molecules should be pulled.
            16,
            576,
        ),
        (TorsionDriveResultCollection, "TorsionDrive Paper", "default", 1, 3),
    ],
)
def test_collection_from_server(
    collection_type, dataset, spec, n_molecules, n_results, public_client
):
    """Test downloading a dataset from the QCArchive."""

    result = collection_type.from_server(
        client=public_client,
        datasets=dataset,
        spec_name=spec,
    )

    assert public_client.address in result.entries

    assert result.n_molecules == n_molecules
    assert result.n_results == n_results


@pytest.mark.parametrize(
    "collection, record",
    [
        (
            BasicResultCollection(
                entries={
                    "https://api.qcarchive.molssi.org:443/": [
                        BasicResult(
                            record_id=ObjectId("1"),
                            cmiles="[He:1]",
                            inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                        )
                    ]
                }
            ),
            ResultRecord(
                program="psi4",
                driver=DriverEnum.gradient,
                method="scf",
                basis="sto-3g",
                molecule=ObjectId("1"),
                status=RecordStatusEnum.complete,
            ),
        ),
        (
            OptimizationResultCollection(
                entries={
                    "https://api.qcarchive.molssi.org:443/": [
                        OptimizationResult(
                            record_id=ObjectId("1"),
                            cmiles="[He:1]",
                            inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                        )
                    ]
                }
            ),
            OptimizationRecord(
                program="psi4",
                qc_spec=QCSpecification(
                    driver=DriverEnum.gradient,
                    method="scf",
                    basis="sto-3g",
                    program="psi4",
                ),
                initial_molecule=ObjectId("1"),
                status=RecordStatusEnum.complete,
            ),
        ),
    ],
)
def test_to_results(collection, record, monkeypatch):
    def mock_query_procedures(*args, **kwargs):
        return [record]

    monkeypatch.setattr(FractalClient, "query_procedures", mock_query_procedures)

    records = collection.to_records()

    assert len(records) == 1
    assert isinstance(records[0], record.__class__)


# def test_optimization_create_basic_dataset(public_client):
#     """
#     Test creating a new basicdataset from the result of an optimization dataset.
#     """
#     result = OptimizationCollectionResult.from_server(
#         client=public_client,
#         spec_name="default",
#         dataset_name="OpenFF Gen 2 Opt Set 1 Roche",
#         include_trajectory=True,
#         final_molecule_only=False,
#         subset=["cn1c(cccc1=o)c2ccccc2oc-0"])
#     new_dataset = result.create_basic_dataset(dataset_name="new basicdataset",
#                                               description="test new optimizationdataset",
#                                               tagline='new optimization dataset',
#                                               driver="energy")
#     assert new_dataset.dataset_name == "new basicdataset"
#     assert new_dataset.n_molecules == 1
#     assert new_dataset.n_records == 1
#     result_geom = result.collection["Cn1c(cccc1=O)c2ccccc2OC"].entries[0].final_molecule.molecule.geometry
#     # make sure the geometry is correct
#     assert new_dataset.dataset["cn1c(cccc1=o)c2ccccc2oc-0"].initial_molecules[0].geometry.all() == result_geom.all()
#
#
# def test_torsion_drive_create_optimization_dataset(public_client):
#     """
#     Tast making a new optimization dataset of constrained optimizations from the results of a torsiondrive dataset.
#     """
#
#     result = TorsionDriveCollectionResult.from_server(client=public_client,
#                                                       spec_name="default",
#                                                       dataset_name="TorsionDrive Paper",
#                                                       include_trajectory=True,
#                                                       final_molecule_only=False,
#                                                       subset=["[ch2:3]([ch2:2][oh:4])[oh:1]_12"])
#     # make a new torsiondrive dataset
#     new_dataset = result.create_optimization_dataset(dataset_name="new optimization dataset",
#                                                      description="a test optimization dataset",
#                                                      tagline="a test optimization dataset.")
#
#     assert new_dataset.dataset_name == "new optimization dataset"
#     assert new_dataset.n_molecules == 1
#     assert new_dataset.n_records == 24
#     dihedrals = set()
#     for entry in new_dataset.dataset.values():
#         assert entry.constraints.has_constraints is True
#         assert len(entry.constraints.set) == 1
#         dihedrals.add(entry.constraints.set[0].value)
#
#     # now sort the dihedrals and make sure they are all present
#     dihs = sorted(dihedrals)
#     refs = [x for x in range(-165, 195, 15)]
#     assert dihs == refs
