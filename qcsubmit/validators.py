"""
Centralise the validators for easy reuse between factories and datasets.
"""

from typing import Dict, Tuple

import openforcefield.topology as off
import qcelemental as qcel

from .exceptions import (
    DatasetInputError,
    DihedralConnectionError,
    LinearTorsionError,
    MolecularComplexError,
)


def cmiles_validator(cmiles: Dict[str, str]) -> Dict[str, str]:
    """
    Validate the cmiles attributes for a molecule submission.

    Parameters:
        cmiles: The cmiles attributes for the molecule which is to be submitted.

    Raises:
        DatasetInputError: If the cmiles is missing a field.
    """

    expected_cmiles = {
        "canonical_smiles",
        "canonical_isomeric_smiles",
        "canonical_explicit_hydrogen_smiles",
        "canonical_isomeric_explicit_hydrogen_smiles",
        "canonical_isomeric_explicit_hydrogen_mapped_smiles",
        "molecular_formula",
        "standard_inchi",
        "inchi_key",
    }
    # use set logic to find missing expected attributes from cmiles
    supplied_cmiles = set(cmiles.keys())
    difference = expected_cmiles.difference(supplied_cmiles)
    if difference:
        raise DatasetInputError(
            f"The supplied cmiles is missing the following fields {difference}."
        )

    return cmiles


def scf_property_validator(scf_property: str) -> str:
    """
    Validate a single scf property this is used for each in a list and also for adding  a new property.

    Parameters:
        scf_property: The scf property which is to be added.

    Raises:
        DatasetInputError: If the scf property is not correct.
    """

    allowed_properties = [
        "dipole",
        "quadrupole",
        "mulliken_charges",
        "lowdin_charges",
        "wiberg_lowdin_indices",
        "mayer_indices",
    ]

    if scf_property.lower() not in allowed_properties:
        raise DatasetInputError(
            f"The requested scf_property {scf_property} is not valid please chose from {allowed_properties}."
        )

    return scf_property.lower()


def check_improper_connection(
    improper: Tuple[int, int, int, int], molecule: off.Molecule
) -> Tuple[int, int, int, int]:
    """
    Check that the given improper is part of the molecule, this makes sure that all atoms are connected to the
    central atom.

    Parameters:
        improper: The imporper torsion that should be checked.
        molecule: The molecule which we want to check the improper in.

    Returns:
        The validated improper torsion tuple.

    Raises:
        DihedralConnectionError: If the improper dihedral is not valid on this molecule.
    """

    for atom_index in improper:
        atom = molecule.atoms[atom_index]
        bonded_atoms = set()
        for neighbour in atom.bonded_atoms:
            bonded_atoms.add(neighbour.molecule_atom_index)
        # if the set has three common atoms this is the central atom of an improper
        if len(bonded_atoms.intersection(set(improper))) == 3:
            return improper

    raise DihedralConnectionError(
        f"The given improper dihedral {improper} was not valid for molecule {molecule}."
    )


def check_torsion_connection(
    torsion: Tuple[int, int, int, int], molecule: off.Molecule
) -> Tuple[int, int, int, int]:
    """
    Check that the given torsion indices create a connected torsion in the molecule.

    Parameters:
        torsion: The torsion indices that should be checked.
        molecule: The molecule which we want to check the torsion in.

    Returns:
        The validated torsion tuple.

    Raises:
        DihedralConnectionError: If the proper dihedral is not valid for this molecule.
    """

    for i in range(3):
        # get the atoms to be checked
        atoms = [torsion[i], torsion[i + 1]]
        try:
            _ = molecule.get_bond_between(*atoms)
        except (off.topology.NotBondedError, IndexError):
            # catch both notbonded errors and tags on atoms not in the molecule
            raise DihedralConnectionError(
                f"The dihedral {torsion} was not valid for the molecule {molecule}, as there is no bond between atoms {atoms}."
            )

    return torsion


def check_linear_torsions(
    torsion: Tuple[int, int, int, int], molecule: off.Molecule
) -> Tuple[int, int, int, int]:
    """
    Check that the torsion supplied is not for a linear bond.

    Parameters:
        torsion: The indices of the atoms in the selected torsion.
        molecule: The molecule which should be checked.

    Raises:
        LinearTorsionError: If the given torsion involves driving a linear bond.
    """

    # this is based on the past submissions to QCarchive which have failed
    # highlight the central bond of a linear torsion
    linear_smarts = "[*!D1:1]~[$(*#*)&D2,$(C=*)&D2:2]"

    matches = molecule.chemical_environment_matches(linear_smarts)

    if torsion[1:3] in matches or torsion[2:0:-1] in matches:
        raise LinearTorsionError(
            f"The dihedral {torsion} in molecule {molecule} highlights a linear bond."
        )

    return torsion


def check_valence_connectivity(molecule: qcel.models.Molecule) -> qcel.models.Molecule:
    """
    Check if the given molecule is one single molecule, also warn about imcomplete valence.
    """

    import warnings

    if molecule.molecular_charge != 0:
        warnings.warn(
            f"The molecule {molecule.name} has a net charge of {molecule.molecular_charge}.",
            UserWarning,
        )

    if len(molecule.fragment_charges) > 1:
        raise MolecularComplexError(
            f"The molecule {molecule.name} is a complex made of {len(molecule.fragment_charges)} units."
        )

    return molecule
