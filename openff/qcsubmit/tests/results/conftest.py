import numpy as np
import pandas as pd
import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit
from qcportal import PortalClient
#from qcportal.collections import OptimizationDataset
#from qcportal.collections.optimization_dataset import OptEntry
#from qcportal.models.records import OptimizationRecord
from qcportal.optimization import (OptimizationDataset, OptimizationDatasetEntry as OptEntry, OptimizationRecord)

from openff.qcsubmit.results import (
    BasicResultCollection,
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.tests.results import (
    mock_basic_result_collection,
    mock_optimization_result_collection,
    mock_torsion_drive_result_collection,
)


def _smiles_to_molecule(smiles: str) -> Molecule:

    molecule: Molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)

    return molecule


@pytest.fixture()
def basic_result_collection(monkeypatch) -> BasicResultCollection:
    """Create a basic collection which can be filtered."""

    smiles = {
        "http://localhost:442": [
            _smiles_to_molecule(smiles) for smiles in ["CO", "CCO", "CCO", "CCCO"]
        ],
        "http://localhost:443": [
            _smiles_to_molecule(smiles) for smiles in ["C=O", "CC=O", "CC=O", "CCC=O"]
        ],
    }

    return mock_basic_result_collection(smiles, monkeypatch)


@pytest.fixture()
def tautomer_basic_result_collection(monkeypatch) -> BasicResultCollection:
    """Create a basic result collection with tautomers."""

    smiles = {
        "http://localhost:442": [
            _smiles_to_molecule(smiles) for smiles in ["Oc1nnccn1", "C1=NC(=O)NN=C1", "C1=CN=NC(=O)N1"]
        ]
    }
    return mock_basic_result_collection(smiles, monkeypatch)


@pytest.fixture()
def h_bond_basic_result_collection(monkeypatch) -> BasicResultCollection:
    """Create a basic collection which can be filtered."""

    # Create a molecule which contains internal h-bonds.
    h_bond_molecule = Molecule.from_smiles(r"O\C=C/C=O")
    h_bond_molecule.add_conformer(
        np.array(
            [
                [0.5099324, -1.93893933, 0.62593794],
                [-0.11678398, -0.78811455, 0.23294619],
                [0.54772449, 0.32974607, -0.06212188],
                [2.01855326, 0.32851657, 0.03836611],
                [2.68037677, -0.68459523, -0.15270394],
                [1.47464514, -1.82358289, 0.65648550],
                [-1.1913352, -0.90038794, 0.19441436],
                [0.04801793, 1.23909473, -0.37244973],
                [2.49137521, 1.29117954, 0.29548031],
            ]
        )
        * unit.angstrom
    )

    smiles = {
        "http://localhost:442": [h_bond_molecule],
        "http://localhost:443": [_smiles_to_molecule("CO")],
    }

    return mock_basic_result_collection(smiles, monkeypatch)


@pytest.fixture()
def optimization_result_collection(monkeypatch) -> OptimizationResultCollection:
    """Create a basic collection which can be filtered."""

    # Create a molecule which contains internal h-bonds.
    h_bond_molecule = Molecule.from_smiles(r"O\C=C/C=O")
    h_bond_molecule.add_conformer(
        np.array(
            [
                [0.5099324, -1.93893933, 0.62593794],
                [-0.11678398, -0.78811455, 0.23294619],
                [0.54772449, 0.32974607, -0.06212188],
                [2.01855326, 0.32851657, 0.03836611],
                [2.68037677, -0.68459523, -0.15270394],
                [1.47464514, -1.82358289, 0.65648550],
                [-1.1913352, -0.90038794, 0.19441436],
                [0.04801793, 1.23909473, -0.37244973],
                [2.49137521, 1.29117954, 0.29548031],
            ]
        )
        * unit.angstrom
    )

    smiles = {
        "http://localhost:442": [
            _smiles_to_molecule(smiles) for smiles in ["CO", "CCO", "CCO", "CCCO"]
        ],
        "http://localhost:443": [h_bond_molecule],
    }

    return mock_optimization_result_collection(smiles, monkeypatch)


@pytest.fixture()
def optimization_result_collection_duplicates(monkeypatch) -> OptimizationResultCollection:
    """Create a collection with duplicate enetries accross different addresses which can be reduced to a single entry."""

    smiles = {
        "http://localhost:442": [
            _smiles_to_molecule(smiles="CCCO")
        ],
        "http://localhost:443": [
            _smiles_to_molecule(smiles="CCCO")
        ]
    }
    return mock_optimization_result_collection(smiles, monkeypatch)


@pytest.fixture()
def torsion_drive_result_collection(monkeypatch) -> TorsionDriveResultCollection:
    """Create a basic collection which can be filtered."""

    # Create a molecule which contains atleast one internal h-bond.
    h_bond_molecule = Molecule.from_smiles(r"O\C=C/C=O")
    h_bond_molecule.add_conformer(
        np.array(
            [
                [0.5099324, -1.93893933, 0.62593794],
                [-0.11678398, -0.78811455, 0.23294619],
                [0.54772449, 0.32974607, -0.06212188],
                [2.01855326, 0.32851657, 0.03836611],
                [2.68657255, 1.32275689, -0.21883053],
                [-0.13074179, -2.64932013, 0.79858345],
                [-1.1913352, -0.90038794, 0.19441436],
                [0.04801793, 1.23909473, -0.37244973],
                [2.48549175, -0.61807805, 0.35823375],
            ]
        )
        * unit.angstrom
    )
    h_bond_molecule.add_conformer(
        np.array(
            [
                [0.5099324, -1.93893933, 0.62593794],
                [-0.11678398, -0.78811455, 0.23294619],
                [0.54772449, 0.32974607, -0.06212188],
                [2.01855326, 0.32851657, 0.03836611],
                [2.68037677, -0.68459523, -0.15270394],
                [1.47464514, -1.82358289, 0.65648550],
                [-1.1913352, -0.90038794, 0.19441436],
                [0.04801793, 1.23909473, -0.37244973],
                [2.49137521, 1.29117954, 0.29548031],
            ]
        )
        * unit.angstrom
    )

    # Create a molecule which contains no internal h-bonds.
    no_h_bond_molecule = Molecule.from_smiles(r"O\C=C/C=O")
    no_h_bond_molecule.add_conformer(h_bond_molecule.conformers[0])
    no_h_bond_molecule.add_conformer(
        h_bond_molecule.conformers[0] + 1.0 * unit.angstrom
    )

    smiles = {
        "http://localhost:442": [h_bond_molecule],
        "http://localhost:443": [no_h_bond_molecule],
    }

    return mock_torsion_drive_result_collection(smiles, monkeypatch)


@pytest.fixture()
def optimization_dataset_invalid_cmiles(monkeypatch, fractal_compute_server):
    """Creates a mocked qcportal optimization dataset with one normal and one invalid cmiles record."""
    client = PortalClient(fractal_compute_server)
    # Fake the records in the dataset with missing data
    data = {
        "GNT-00284-0": OptEntry(
            name="GNT-00284-0",
            initial_molecule="33240545",
            additional_keywords={},
            attributes={
                "canonical_isomeric_explicit_hydrogen_mapped_smiles": "[F:1][c:2]1[c:3]([H:32])[c:4]([H:33])[c:5]([H:34])[c:6]([F:7])[c:8]1[C:9]1=[N:12][N:13]2[C:14](=[C:15]([H:37])[N:16]=[C:17]2[N:18]([c:19]2[c:20]([H:39])[nH+:21][c:22]([H:40])[c:23]([H:41])[c:24]2[N:25]2[C:26]([H:42])([H:43])[C@:30]([NH+:31]([H:51])[H:52])([H:50])[C:29]([H:48])([H:49])[C:28]([H:46])([H:47])[C:27]2([H:44])[H:45])[H:38])[C:11]([H:36])=[C:10]1[H:35]",
                "molecular_formula": "C22H21F2N7",
                "standard_inchi": "InChI=1S/C22H21F2N7/c23-16-4-1-5-17(24)21(16)18-7-6-15-11-27-22(31(15)29-18)28-19-12-26-9-8-20(19)30-10-2-3-14(25)13-30/h1,4-9,11-12,14H,2-3,10,13,25H2,(H,27,28)/p+2/t14-/m0/s1",
                "inchi_key": "MLPJIYXHHVPOPK-AWEZNQCLSA-P",
            },
            object_map={"default": "1"},
        ),
        "BRI-01356-0": OptEntry(
            name="BRI-01356-0",
            initial_molecule="33222549",
            additional_keywords={},
            attributes={
                "canonical_isomeric_explicit_hydrogen_mapped_smiles": "[N:1]([C:2](=[O:3])[c:4]1[c:5]([H:26])[c:6]([H:27])[c:7]([N:8]([C:9](=[O:10])[C:11]([N:12]2[C:13]([H:31])([H:32])[C:14]([H:33])([H:34])[C:15]([N:18]([C:19]([O:20][H:41])=[O:21])[H:40])([H:35])[C:16]([H:36])([H:37])[C:17]2([H:38])[H:39])([H:29])[H:30])[H:28])[c:22]([H:42])[c:23]1[H:43])([H:24])[H:25]",
                "molecular_formula": "C15H20N4O4",
                "standard_inchi": "InChI=1S/C15H20N4O4/c16-14(21)10-1-3-11(4-2-10)17-13(20)9-19-7-5-12(6-8-19)18-15(22)23/h1-4,12,18H,5-9H2,(H2,16,21)(H,17,20)(H,22,23)",
                "inchi_key": "AZHQPEOBSZLCKD-UHFFFAOYSA-N",
            },
            object_map={"default": "2"},
        ),
    }
    dataset = OptimizationDataset(
        name="invalid-collection", client=client, records=data
    )

    # now mock the query function which is called when processing the results
    def query(spec_name):
        mock_spec = {
            "driver": "gradient",
            "method": "B3LYP-D3BJ",
            "basis": "DZVP",
            "program": "psi4",
        }
        results = {
            opt.name: OptimizationRecord(
                id=opt.object_map["default"],
                procedure="optimization",
                program="geometric",
                version=1,
                status="COMPLETE",
                initial_molecule="1",
                qc_spec=mock_spec,
            )
            for opt in data.values()
        }
        return pd.Series(results)

    # mock the status of the records
    monkeypatch.setattr(dataset, "query", query)

    return dataset
