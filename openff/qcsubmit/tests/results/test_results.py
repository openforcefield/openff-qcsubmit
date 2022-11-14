"""
Test the results packages when collecting from the public qcarchive.
"""

import numpy
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from pydantic import ValidationError
from qcportal import FractalClient
from qcportal.models import Molecule as QCMolecule
from qcportal.models import (
    ObjectId,
    OptimizationRecord,
    ResultRecord,
    TorsionDriveRecord,
)
from qcportal.models.common_models import (
    DriverEnum,
    OptimizationSpecification,
    QCSpecification,
)
from qcportal.models.records import RecordStatusEnum
from qcportal.models.torsiondrive import TDKeywords

from openff.qcsubmit.common_structures import QCSpec
from openff.qcsubmit.exceptions import RecordTypeError
from openff.qcsubmit.results import (
    BasicResult,
    BasicResultCollection,
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.results.filters import ResultFilter
from openff.qcsubmit.results.results import (
    OptimizationResult,
    TorsionDriveResult,
    _BaseResultCollection,
)
from openff.qcsubmit.tests import does_not_raise


class MockServerInfo:

    def dict(self):
        return {
            "name": "Mock",
            "query_limit": 2000,
            "version": "0.0.0",
            "client_lower_version_limit": "0.0.0",
            "client_upper_version_limit": "10.0.0",
        }


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


def test_base_filter(basic_result_collection):
    class DummyFilter(ResultFilter):
        def _apply(self, result_collection):

            result_collection.entries = {
                "http://localhost:442": result_collection.entries[
                    "http://localhost:442"
                ]
            }

            return result_collection

    filtered_collection = basic_result_collection.filter(
        DummyFilter(),
        DummyFilter(),
    )

    assert filtered_collection.n_results == 4
    assert filtered_collection.n_molecules == 3

    assert "applied-filters" in filtered_collection.provenance
    assert "DummyFilter-0" in filtered_collection.provenance["applied-filters"]
    assert "DummyFilter-1" in filtered_collection.provenance["applied-filters"]


def test_base_smirnoff_coverage():

    collection = TorsionDriveResultCollection(
        entries={
            "http://localhost:442": [
                TorsionDriveResult(
                    record_id=ObjectId(str(i + 1)),
                    cmiles=smiles,
                    inchi_key=Molecule.from_smiles(smiles).to_inchikey(),
                )
                for i, smiles in enumerate(
                    [
                        "[H:1][C:2]([H:3])([H:4])([H:5])",
                        "[C:1]([H:2])([H:3])([H:4])([H:5])",
                    ]
                )
            ]
        }
    )

    coverage = collection.smirnoff_coverage(ForceField("openff-1.3.0.offxml"))

    assert {*coverage} == {"Bonds", "Angles", "vdW", "Constraints"}
    assert all(count == 1 for counts in coverage.values() for count in counts.values())


@pytest.mark.parametrize(
    "collection_type, dataset, spec, n_molecules, n_results",
    [
        (
            BasicResultCollection,
            "OpenFF BCC Refit Study COH v1.0",
            "resp-2-vacuum",
            91,
            191,
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
                            cmiles="[Cl:1][Cl:2]",
                            inchi_key="KZBUYRJDOAKODT-UHFFFAOYSA-N",
                        )
                    ]
                }
            ),
            ResultRecord(
                id=ObjectId("1"),
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
                            cmiles="[Cl:1][Cl:2]",
                            inchi_key="KZBUYRJDOAKODT-UHFFFAOYSA-N",
                        )
                    ]
                }
            ),
            OptimizationRecord(
                program="psi4",
                id=ObjectId("1"),
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
        (
            TorsionDriveResultCollection(
                entries={
                    "https://api.qcarchive.molssi.org:443/": [
                        TorsionDriveResult(
                            record_id=ObjectId("1"),
                            cmiles="[Cl:1][Cl:2]",
                            inchi_key="KZBUYRJDOAKODT-UHFFFAOYSA-N",
                        )
                    ]
                }
            ),
            TorsionDriveRecord(
                id=ObjectId("1"),
                qc_spec=QCSpecification(
                    driver=DriverEnum.gradient,
                    method="scf",
                    basis="sto-3g",
                    program="psi4",
                ),
                optimization_spec=OptimizationSpecification(
                    program="geometric", keywords={}
                ),
                initial_molecule=[ObjectId("1")],
                status=RecordStatusEnum.complete,
                keywords=TDKeywords(dihedrals=[], grid_spacing=[]),
                final_energy_dict={},
                optimization_history={},
                minimum_positions={},
            ),
        ),
    ],
)
def test_to_records(collection, record, monkeypatch):
    def mock_query_procedures(*args, **kwargs):
        return [record]

    def mock_query_molecules(*args, **kwargs):

        molecule: Molecule = Molecule.from_smiles("[Cl:1][Cl:2]")
        molecule.add_conformer(numpy.arange(6).reshape((2, 3)) * unit.angstrom)

        qc_molecule = QCMolecule(
            **molecule.to_qcschema().dict(exclude={"id"}), id=args[1][0]
        )

        return [qc_molecule]

    monkeypatch.setattr(FractalClient, "query_procedures", mock_query_procedures)
    monkeypatch.setattr(FractalClient, "query_molecules", mock_query_molecules)

    records_and_molecules = collection.to_records()
    assert len(records_and_molecules) == 1

    record, molecule = records_and_molecules[0]

    assert isinstance(record, record.__class__)

    if not isinstance(record, TorsionDriveRecord):
        assert molecule.n_conformers == 1


def test_optimization_create_basic_dataset(optimization_result_collection):
    """
    Test creating a new ``BasicDataset`` from the result of an optimization dataset.
    """

    dataset = optimization_result_collection.create_basic_dataset(
        dataset_name="new basicdataset",
        description="test new optimizationdataset",
        tagline='new optimization dataset',
        driver="energy",
        qc_specs=[QCSpec(spec_name="some-name", basis="6-31G")]
    )

    assert len(dataset.qc_specifications) == 1
    assert {*dataset.qc_specifications} == {"some-name"}
    assert dataset.qc_specifications["some-name"].basis == "6-31G"

    assert dataset.dataset_name == "new basicdataset"
    assert dataset.n_molecules == 4
    assert dataset.n_records == 5  # the collection contains 1 duplicate

    
def test_optimization_to_basic_result_collection(
    optimization_result_collection, monkeypatch
):

    def mock_automodel_request(*args, **kwargs):
        return MockServerInfo()

    def mock_query_results(*args, **kwargs):

        assert "program" in kwargs and kwargs["program"] == "psi4"
        assert "method" in kwargs and kwargs["method"] == "scf"
        assert "basis" in kwargs and kwargs["basis"] == "sto-3g"
        assert "driver" in kwargs and kwargs["driver"] == "hessian"

        return [
            ResultRecord(
                id=ObjectId("1"),
                program=kwargs["program"],
                driver=getattr(DriverEnum, kwargs["driver"]),
                method=kwargs["method"],
                basis=kwargs["basis"],
                molecule=kwargs["molecule"][0],
                status=RecordStatusEnum.complete,
            )
        ]

    monkeypatch.setattr(FractalClient, "_automodel_request", mock_automodel_request)
    monkeypatch.setattr(FractalClient, "query_results", mock_query_results)

    basic_collection = optimization_result_collection.to_basic_result_collection(
        "hessian"
    )

    assert basic_collection.n_results == 2
    assert basic_collection.n_molecules == 2


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


def test_torsion_smirnoff_coverage(public_client, monkeypatch):

    molecule: Molecule = Molecule.from_mapped_smiles(
        "[H:1][C:2]([H:7])([H:8])"
        "[C:3]([H:9])([H:10])"
        "[C:4]([H:11])([H:12])"
        "[C:5]([H:13])([H:14])[H:6]"
    )

    dihedrals = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]

    collection = TorsionDriveResultCollection(
        entries={
            "http://localhost:442": [
                TorsionDriveResult(
                    record_id=ObjectId(str(i + 1)),
                    cmiles=molecule.to_smiles(mapped=True),
                    inchi_key=molecule.to_inchikey(),
                )
                for i in range(len(dihedrals))
            ]
        }
    )

    monkeypatch.setattr(
        TorsionDriveResultCollection,
        "to_records",
        lambda self: [
            (
                TorsionDriveRecord(
                    id=entry.record_id,
                    qc_spec=QCSpecification(
                        driver=DriverEnum.gradient,
                        method="scf",
                        basis="sto-3g",
                        program="psi4",
                    ),
                    optimization_spec=OptimizationSpecification(
                        program="geometric", keywords={}
                    ),
                    initial_molecule=[ObjectId(1)],
                    status=RecordStatusEnum.complete,
                    client=public_client,
                    keywords=TDKeywords(
                        dihedrals=[dihedrals[int(entry.record_id) - 1]], grid_spacing=[]
                    ),
                    final_energy_dict={},
                    optimization_history={},
                    minimum_positions={},
                ),
                molecule,
            )
            for address, entries in self.entries.items()
            for entry in entries
        ],
    )

    coverage = collection.smirnoff_coverage(
        ForceField("openff-1.3.0.offxml"), driven_only=True
    )

    assert {*coverage} == {"Bonds", "Angles", "ProperTorsions"}

    assert {*coverage["Bonds"].values()} == {3}
    assert {*coverage["Angles"].values()} == {3}

    assert {*coverage["ProperTorsions"].values()} == {1, 3}
