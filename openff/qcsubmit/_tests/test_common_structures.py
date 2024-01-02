import pytest
from openff.toolkit.topology import Molecule

from openff.qcsubmit._tests import does_not_raise
from openff.qcsubmit.common_structures import Metadata, MoleculeAttributes, QCSpec
from openff.qcsubmit.exceptions import DatasetInputError, QCSpecificationError


def test_attributes_from_openff_molecule():
    mol = Molecule.from_smiles("CC")

    attributes = MoleculeAttributes.from_openff_molecule(mol)

    # now make our own cmiles
    test_cmiles = {
        "canonical_smiles": mol.to_smiles(
            isomeric=False, explicit_hydrogens=False, mapped=False
        ),
        "canonical_isomeric_smiles": mol.to_smiles(
            isomeric=True, explicit_hydrogens=False, mapped=False
        ),
        "canonical_explicit_hydrogen_smiles": mol.to_smiles(
            isomeric=False, explicit_hydrogens=True, mapped=False
        ),
        "canonical_isomeric_explicit_hydrogen_smiles": mol.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=False
        ),
        "canonical_isomeric_explicit_hydrogen_mapped_smiles": mol.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=True
        ),
        "molecular_formula": mol.hill_formula,
        "standard_inchi": mol.to_inchi(fixed_hydrogens=False),
        "inchi_key": mol.to_inchikey(fixed_hydrogens=False),
        "fixed_hydrogen_inchi": mol.to_inchi(fixed_hydrogens=True),
        "fixed_hydrogen_inchi_key": mol.to_inchikey(fixed_hydrogens=True),
        "unique_fixed_hydrogen_inchi_keys": {mol.to_inchikey(fixed_hydrogens=True)},
    }
    assert test_cmiles == attributes


def test_attributes_from_openff_with_map():
    """
    Make sure we can provide a valid cmiles for a molecule with an atom_map and ensure the map is not removed.
    """

    mol = Molecule.from_smiles("CC")
    atom_map = {0: 0, 1: 1, 2: 2, 3: 3}
    mol.properties["atom_map"] = atom_map
    cmiles = MoleculeAttributes.from_openff_molecule(molecule=mol)
    assert "atom_map" in mol.properties
    _ = cmiles.to_openff_molecule()


def test_attributes_from_openff_multi_component():
    """
    Make sure the unique inchi keys are updated correctly.
    """
    mol = Molecule.from_smiles(
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5.CS(=O)(=O)O"
    )

    attributes = MoleculeAttributes.from_openff_molecule(mol)
    assert len(attributes.unique_fixed_hydrogen_inchi_keys) == 2


def test_attributes_to_openff_molecule():
    """Round trip a molecule to and from its attributes."""

    mol: Molecule = Molecule.from_smiles("CC")

    attributes = MoleculeAttributes.from_openff_molecule(molecule=mol)

    mol2 = attributes.to_openff_molecule()

    isomorphic, atom_map = Molecule.are_isomorphic(mol, mol2, return_atom_map=True)
    assert isomorphic is True
    # make sure the molecules are in the same order
    assert atom_map == dict((i, i) for i in range(mol.n_atoms))


@pytest.mark.parametrize(
    "metadata, expected_raises",
    [
        (
            Metadata(
                collection_type="torsiondrive",
                dataset_name="ABC",
                short_description="abcdefgh",
                long_description_url="https://github.com/openforcefield",
                long_description="abcdefgh",
                elements={"C", "H"},
            ),
            does_not_raise(),
        ),
        (
            Metadata(
                collection_type="torsiondrive",
                dataset_name="ABC",
                short_description="abcdefgh",
                long_description="abcdefgh",
                elements={"C", "H"},
            ),
            does_not_raise(),
        ),
        (
            Metadata(),
            pytest.raises(
                DatasetInputError,
                match="The metadata has the following incomplete fields",
            ),
        ),
    ],
)
def test_validate_metadata(metadata, expected_raises):
    with expected_raises:
        metadata.validate_metadata(raise_errors=True)


def test_scf_prop_validation():
    """
    Make sure unsupported scf properties are not allowed into a QCSpec.
    """

    with pytest.raises(QCSpecificationError):
        QCSpec(scf_properties=["ddec_charges"])
