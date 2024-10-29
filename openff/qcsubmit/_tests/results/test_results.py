"""
Test the results packages when collecting from the public qcarchive.
"""

import datetime
from tempfile import TemporaryDirectory

import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from qcelemental.models import DriverEnum
from qcportal.torsiondrive import (
    TorsiondriveKeywords,
    TorsiondriveRecord,
    TorsiondriveSpecification,
)

from openff.qcsubmit._pydantic import ValidationError
from openff.qcsubmit._tests import does_not_raise
from openff.qcsubmit.common_structures import QCSpec
from openff.qcsubmit.exceptions import RecordTypeError
from openff.qcsubmit.results import (
    BasicResult,
    BasicResultCollection,
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.results.filters import ResultFilter
from openff.qcsubmit.results.results import TorsionDriveResult, _BaseResultCollection
from openff.qcsubmit.utils import _CachedPortalClient, portal_client_manager

from . import (
    OptimizationRecord,
    OptimizationSpecification,
    QCSpecification,
    RecordStatusEnum,
    SinglepointRecord,
)


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
        record_id=1,
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
                    record_id=1,
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                )
            ],
            does_not_raise(),
        ),
        (
            [
                BasicResult(
                    record_id=1,
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
                    record_id=1,
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                )
            ],
            "http://localhost:443": [
                BasicResult(
                    record_id=1,
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
                    record_id=1,
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                )
            ],
            "http://localhost:443": [
                BasicResult(
                    record_id=1,
                    cmiles="[He:1]",
                    inchi_key="SWQJXJOGLNCZEY-UHFFFAOYSA-N",
                ),
                BasicResult(
                    record_id=2,
                    cmiles="[Cl:1][Cl:2]",
                    inchi_key="KZBUYRJDOAKODT-UHFFFAOYSA-N",
                ),
            ],
        }
    )

    assert collection.n_molecules == 2


def test_base_validate_record_types():
    records = [
        SinglepointRecord(
            # SinglepointRecord(
            specification=QCSpecification(
                program="psi4",
                driver=DriverEnum.gradient,
                method="scf",
                basis="sto-3g",
            ),
            molecule_id=1,
            status=RecordStatusEnum.complete,
            is_service=False,
            created_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
            modified_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
            id=1,
        ),
        OptimizationRecord(
            # OptimizationRecord(
            specification=OptimizationSpecification(
                program="geometric",
                qc_specification=QCSpecification(
                    driver=DriverEnum.gradient,
                    method="scf",
                    basis="sto-3g",
                    program="psi4",
                ),
            ),
            initial_molecule_id=1,
            status=RecordStatusEnum.complete,
            is_service=False,
            created_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
            modified_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
            id=1,
        ),
    ]

    _BaseResultCollection._validate_record_types(records[:1], SinglepointRecord)
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
                    record_id=i + 1,
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
            # "resp-2-vacuum",
            "spec_2",
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
    "collection_name, collection_type, spec_name, expected_n_recs, expected_n_mols",
    [
        # spec_2 corresponds to "default" while spec_1 is basically "default with pcm".
        # In spec_1, one calc has failed, so there's only 190 results, but in spec_2 there are 191.
        ("OpenFF BCC Refit Study COH v1.0", BasicResultCollection, "spec_2", 191, 91),
        (
            "OpenFF Gen 2 Opt Set 3 Pfizer Discrepancy",
            OptimizationResultCollection,
            "default",
            197,
            49,
        ),
        # ("OpenFF Protein Capped 3-mer Omega v1.0", TorsionDriveResultCollection, "default", 24, 24), #26 total, 2 failed
        # ("OpenFF DANCE 1 eMolecules t142 v1.0", TorsionDriveResultCollection, "default", 20, 20),
        # ("OpenFF Gen 2 Torsion Set 6 Supplemental", TorsionDriveResultCollection, "default", 7, 7)
        # ("OpenFF Protein Dipeptide 2-D TorsionDrive v1.1", TorsionDriveResultCollection, "default", 5, 5)
        (
            "OpenFF Group1 Torsions 3",
            TorsionDriveResultCollection,
            "default",
            5,
            5,
        ),  # 6 total, 1 failed
    ],
)
def test_to_records(
    public_client,
    collection_name,
    collection_type,
    spec_name,
    expected_n_recs,
    expected_n_mols,
):
    # as of Aug 2024, this is the one place we're testing caching with
    # _CachedPortalClient and portal_client_manager, which is why the code is
    # so complicated. to_records itself is called in several other tests, where
    # it will use the _default_portal_client (PortalClient), and the assertions
    # at the end of this test expect the same results as a normal PortalClient.
    # Only the middle section of this test is cache-specific.
    collection = collection_type.from_server(
        public_client, collection_name, spec_name=spec_name
    )
    assert collection.n_molecules == expected_n_mols

    def disconnected_client(addr, cache_dir):
        ret = _CachedPortalClient(addr, cache_dir)
        ret._req_session = None
        return ret

    with TemporaryDirectory() as d:
        client = _CachedPortalClient(public_client.address, d)
        with portal_client_manager(lambda _: client):
            with (
                client._no_session(),
                pytest.raises(Exception, match="no attribute 'prepare_request'"),
            ):
                collection.to_records()
            records_and_molecules = collection.to_records()
            # TorsionDriveResultCollection.to_records requires fetching
            # molecules, which cannot currently be cached
            if collection_type is not TorsionDriveResultCollection:
                with client._no_session():
                    assert len(collection.to_records()) == len(records_and_molecules)
                # the previous checks show that the *same* client can access
                # its cache without making new requests. disconnected_client
                # instead shows that a newly-constructed client pointing at the
                # same cache_dir can still access the cache
                with portal_client_manager(lambda addr: disconnected_client(addr, d)):
                    assert len(collection.to_records()) == len(records_and_molecules)

    assert len(records_and_molecules) == expected_n_recs
    record, molecule = records_and_molecules[0]

    assert isinstance(record, record.__class__)

    if not isinstance(record, TorsiondriveRecord):
        assert molecule.n_conformers == 1


def test_optimization_create_basic_dataset():
    """Test creating a new ``BasicDataset`` from the result of an optimization
    dataset, and verify that the molecule hashes match to prevent the creation
    of separate records on QCArchive. See issue #297 for more details.
    """

    # these are three real entries from the "OpenFF Sulfur Optimization
    # Training Coverage Supplement v1.0" dataset used in issue #297. the first
    # fails the round-trip in the previous create_basic_dataset implementation
    # but the next two work with either
    opt = OptimizationResultCollection.parse_raw(
        """
        {
            "entries": {
                "https://api.qcarchive.molssi.org:443/": [
                {
                    "type": "optimization",
                    "record_id": 138340682,
                    "cmiles": "[H:11][c:10]1[c:12]([c:14]([c:16]2[c:7]([c:8]1[H:9])[C:5](=[O:6])[C:3](=[C:1]([S:17]2(=[O:18])=[O:19])[H:2])[Br:4])[H:15])[H:13]",
                    "inchi_key": "UKQYACGPJCUZJN-UHFFFAOYNA-N"
                },
                {
                    "type": "optimization",
                    "record_id": 138341382,
                    "cmiles": "[H:18][C@@:10]([C:11](=[O:12])[O:13][H:14])([C:9]([H:19])([H:20])[C:8]([H:21])([H:22])[S:2](=[N:1][H:23])(=[O:3])[C:4]([H:5])([H:6])[H:7])[N:15]([H:16])[H:17]",
                    "inchi_key": "SXTAYKAGBXMACB-AIXGRKOHNA-N"
                },
                {
                    "type": "optimization",
                    "record_id": 138341384,
                    "cmiles": "[H:18][C@@:10]([C:11](=[O:12])[O:13][H:14])([C:9]([H:19])([H:20])[C:8]([H:21])([H:22])[S:2](=[N:1][H:23])(=[O:3])[C:4]([H:5])([H:6])[H:7])[N:15]([H:16])[H:17]",
                    "inchi_key": "SXTAYKAGBXMACB-AIXGRKOHNA-N"
                }
                ]
            },
            "provenance": {},
            "type": "OptimizationResultCollection"
        }
        """
    )

    opt_hashes = {rec.final_molecule.get_hash() for rec, _mol in opt.to_records()}

    basic = opt.create_basic_dataset(
        "dummy basic dataset name",
        "descdesc",
        "tagtagtag",
        driver="hessian",
        qc_specifications=[QCSpec(spec_name="some-name", basis="6-31G")],
    )

    bas_hashes = {mol.molecule.get_hash() for mol in basic._get_entries()}

    n_results = opt.n_results

    # check for molecule agreement between optimization and basic datasets
    assert len(opt_hashes) == n_results
    assert len(bas_hashes) == n_results
    assert len(opt_hashes & bas_hashes) == n_results

    # check general basic dataset construction
    assert len(basic.qc_specifications) == 1
    assert {*basic.qc_specifications} == {"some-name"}
    assert basic.qc_specifications["some-name"].basis == "6-31G"

    assert basic.dataset_name == "dummy basic dataset name"
    assert basic.n_molecules == 2
    assert basic.n_records == 3  # the collection contains 1 duplicate


def test_optimization_to_basic_result_collection(public_client):
    optimization_result_collection = OptimizationResultCollection.from_server(
        public_client, ["OpenFF Gen 2 Opt Set 3 Pfizer Discrepancy"]
    )
    basic_collection = optimization_result_collection.to_basic_result_collection(
        "hessian"
    )
    assert basic_collection.n_results == 197
    assert basic_collection.n_molecules == 49


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
                    record_id=i + 1,
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
                TorsiondriveRecord(
                    # TorsiondriveRecord.construct(
                    specification=TorsiondriveSpecification(
                        program="torsiondrive",
                        keywords=TorsiondriveKeywords(
                            dihedrals=[dihedrals[int(entry.record_id) - 1]],
                            grid_spacing=[],
                        ),
                        optimization_specification=OptimizationSpecification(
                            program="geometric",
                            keywords={},
                            qc_specification=QCSpecification(
                                driver=DriverEnum.gradient,
                                method="scf",
                                basis="sto-3g",
                                program="psi4",
                            ),
                        ),
                    ),
                    id=entry.record_id,
                    initial_molecules=[],
                    status=RecordStatusEnum.complete,
                    is_service=False,
                    created_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
                    modified_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
                    # compute_history=list(),
                    # )
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
