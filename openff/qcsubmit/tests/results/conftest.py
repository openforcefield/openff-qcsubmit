import numpy as np
import pytest
from openff.toolkit.topology import Molecule
from simtk import unit

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
