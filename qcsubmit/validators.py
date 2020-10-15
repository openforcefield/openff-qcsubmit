"""
Centralise the validators for easy reuse between factories and datasets.
"""

from typing import Dict, List, Tuple

import qcelemental as qcel
from openforcefield import topology as off

from .constraints import Constraints
from .exceptions import (
    AngleConnectionError,
    AtomConnectionError,
    BondConnectionError,
    ConstraintError,
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
        try:
            atom = molecule.atoms[atom_index]
            bonded_atoms = set()
            for neighbour in atom.bonded_atoms:
                bonded_atoms.add(neighbour.molecule_atom_index)
            # if the set has three common atoms this is the central atom of an improper
            if len(bonded_atoms.intersection(set(improper))) == 3:
                return improper
        except IndexError:
            continue
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
    try:
        _ = check_general_connection(connected_atoms=torsion, molecule=molecule)
    except AtomConnectionError as e:
        raise DihedralConnectionError(
            f"The dihedral {torsion} was not valid for the molecule {molecule}, as there is no bond between atoms {e.atoms}."
        )

    return torsion


def check_bond_connection(
    bond: Tuple[int, int], molecule: off.Molecule
) -> Tuple[int, int]:
    """
    Check that the given bond indices create a connected bond in the molecule.

    Parameters:
        bond: The bond indices that should be checked.
        molecule: The molecule which we want to check the bond in.

    Returns:
        The validated bond tuple

    Raises:
        BondConnectionError: If the given tuple is not connected in the molecule.
    """
    try:
        _ = check_general_connection(connected_atoms=bond, molecule=molecule)
    except AtomConnectionError:
        raise BondConnectionError(
            f"The bond {bond} was not valid for the molecule {molecule}, as there is no bond between these atoms."
        )

    return bond


def check_angle_connection(
    angle: Tuple[int, int, int], molecule: off.Molecule
) -> Tuple[int, int, int]:
    """
    Check that the given angle indices create a connected angle in the molecule.

    Parameters:
        angle: The angle indices that should be checked.
        molecule: The molecule which we want to check the angle in.

    Returns:
        The validated angle tuple.

    Raises:
        AngleConnectionError: If the given angle is not connected in the molecule.
    """
    try:
        _ = check_general_connection(connected_atoms=angle, molecule=molecule)
    except AtomConnectionError as e:
        raise AngleConnectionError(
            f"The angle {angle} was not valid for the molecule {molecule}, as there is no bond between atoms {e.atoms}"
        )
    return angle


def check_general_connection(
    connected_atoms: List[int], molecule: off.Molecule
) -> List[int]:
    """
    Check that the list of atoms are all connected in order by explicit bonds in the given molecule.

    Parameters:
        connected_atoms: A list of the atom indices that should be connected in order.
        molecule: The molecule that should be checked for connected atoms.

    Raises:
        AtomConnectionError: If any two of the given list of atoms are not connected.

    Returns:
        The list of validated connected atom indices.
    """
    for i in range(len(connected_atoms) - 1):
        # get the atoms to be checked
        atoms = [connected_atoms[i], connected_atoms[i + 1]]
        try:
            _ = molecule.get_bond_between(*atoms)
        except (off.topology.NotBondedError, IndexError):
            # catch both notbonded errors and tags on atoms not in the molecule
            raise AtomConnectionError(
                f"The set of atoms {connected_atoms} was not valid for the molecule {molecule}, as there is no bond between atoms {atoms}.",
                atoms=atoms,
            )

    return connected_atoms


def check_constraints(constraints: Constraints, molecule: off.Molecule) -> Constraints:
    """
    Warn the user if any of the constraints are between atoms which are not bonded.
    """

    _constraint_to_check = {
        "distance": check_bond_connection,
        "angle": check_angle_connection,
        "dihedral": check_torsion_connection,
    }

    if constraints.has_constraints:
        # check each constraint and warn if it is incorrect
        all_constraints = [
            constraint
            for constraint_set in [constraints.freeze, constraints.set]
            for constraint in constraint_set
        ]
        for constraint in all_constraints:
            if constraint.type != "xyz":
                if constraint.bonded:
                    try:
                        _constraint_to_check[constraint.type](
                            constraint.indices, molecule
                        )
                    except (
                        BondConnectionError,
                        AngleConnectionError,
                        DihedralConnectionError,
                    ) as e:
                        raise ConstraintError(
                            f"The molecule {molecule} has non bonded constraints, if this is intentional add the constraint with the flag `bonded=False`. See error for more details {e.error_message}"
                        )

    return constraints


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
    Check if the given molecule is one single molecule, also warn about incomplete valence.
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
