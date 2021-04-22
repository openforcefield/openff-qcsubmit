import pytest
from openff.toolkit.topology import Molecule
from qcportal.models import ObjectId

from openff.qcsubmit.results import BasicResult, BasicResultCollection


@pytest.fixture()
def basic_result_collection() -> BasicResultCollection:
    """Create a basic collection which can be filtered."""

    return BasicResultCollection(
        entries={
            "http://localhost:442": [
                BasicResult(
                    record_id=ObjectId(str(i + 1)),
                    cmiles=Molecule.from_smiles(smiles).to_smiles(mapped=True),
                    inchi_key=Molecule.from_smiles(smiles).to_inchikey(),
                )
                for i, smiles in enumerate(["CO", "CCO", "CCO", "CCCO"])
            ],
            "http://localhost:443": [
                BasicResult(
                    record_id=ObjectId(str(i + 1)),
                    cmiles=Molecule.from_smiles(smiles).to_smiles(mapped=True),
                    inchi_key=Molecule.from_smiles(smiles).to_inchikey(),
                )
                for i, smiles in enumerate(["C=O", "CC=O", "CC=O", "CCC=O"])
            ],
        }
    )
