import os

import pytest
from openff.toolkit.topology import Molecule

from openff.qcsubmit.utils.visualize import (
    _create_openeye_pdf,
    _create_rdkit_pdf,
    molecules_to_pdf,
)


@pytest.mark.parametrize("create_function", [molecules_to_pdf, _create_openeye_pdf, _create_rdkit_pdf])
def test_create_pdf_function(tmpdir, create_function):
    molecules = [
        Molecule.from_smiles("C"),
        Molecule.from_smiles("CC"),
        Molecule.from_smiles("CCC"),
        Molecule.from_smiles("[H]C#C[H]"),
    ]
    molecules[-1].properties["dihedrals"] = [(0, 1, 2, 3)]

    output_path = os.path.join(tmpdir, "output.pdf")
    create_function(molecules, output_path, 4)

    assert os.path.isfile(output_path)
    assert os.path.getsize(output_path) > 0


def test_molecules_to_pdf_bad_toolkit():
    with pytest.raises(ValueError, match="is not supported, chose"):
        # noinspection PyTypeChecker
        molecules_to_pdf([], "", toolkit="fake-toolkit")
