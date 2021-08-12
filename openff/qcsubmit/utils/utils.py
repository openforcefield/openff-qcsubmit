from typing import Dict, Generator, List, Tuple

from openff.toolkit import topology as off
from openff.toolkit.utils.toolkits import (
    RDKitToolkitWrapper,
    UndefinedStereochemistryError,
)


def get_data(relative_path):
    """
    Get the file path to some data in the qcsubmit package.

    Parameters:
        relative_path: The relative path to the data
    """

    import os

    from pkg_resources import resource_filename

    fn = resource_filename("openff.qcsubmit", os.path.join("data", relative_path))

    if not os.path.exists(fn):
        raise ValueError(
            f"Sorry! {fn} does not exist. If you just added it, you'll have to re-install"
        )

    return fn


def check_missing_stereo(molecule: off.Molecule) -> bool:
    """
    Get if the given molecule has missing stereo by round trip and catching stereo errors.
    Here we use the RDKit backend explicitly for this check as this avoids nitrogen stereochemistry issues with the toolkit.

    Parameters
    ----------
    molecule: off.Molecule
        The molecule which should be checked for stereo issues.

    Returns
    -------
    bool
        `True` if some stereochemistry is missing else `False`.
    """
    try:
        _ = off.Molecule.from_smiles(
            smiles=molecule.to_smiles(isomeric=True, explicit_hydrogens=True),
            hydrogens_are_explicit=True,
            allow_undefined_stereo=False,
            toolkit_registry=RDKitToolkitWrapper(),
        )
        return False
    except UndefinedStereochemistryError:
        return True


def clean_strings(string_list: List[str]) -> List[str]:
    """
    Clean up a list of strings ready to be cast to numbers.
    """
    clean_string = []
    for string in string_list:
        new_string = string.strip()
        clean_string.append(new_string.strip(","))
    return clean_string


def remap_list(target_list: List[int], mapping: Dict[int, int]) -> List[int]:
    """
    Take a list of atom indices and remap them using the given mapping.
    """
    return [mapping[x] for x in target_list]


def condense_molecules(molecules: List[off.Molecule]) -> off.Molecule:
    """
    Take a list of identical molecules in different conformers and collapse them making sure that they are in the same order.
    """
    molecule = molecules.pop()
    for conformer in molecules:
        _, atom_map = off.Molecule.are_isomorphic(
            conformer, molecule, return_atom_map=True
        )
        mapped_mol = conformer.remap(atom_map)
        for geometry in mapped_mol.conformers:
            molecule.add_conformer(geometry)
    return molecule


def chunk_generator(iterable: List, chunk_size: int) -> Generator[List, None, None]:
    """
    Take an iterable and return a list of lists of the specified size.

    Parameters:
         iterable: An iterable object like a list
         chunk_size: The size of each chunk
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def get_torsion(bond: off.Bond) -> Tuple[int, int, int, int]:
    """
    Create a torsion tuple which will be restrained in the torsiondrive.

    Parameters:
        bond: The tuple of the atom indexes for the central bond.

    Returns:
        The tuple of the four atom indices which should be restrained.

    Note:
        If there is more than one possible combination of atoms the heaviest set are selected to be restrained.
    """

    atoms = [bond.atom1, bond.atom2]
    terminal_atoms = {}

    for atom in atoms:
        for neighbour in atom.bonded_atoms:
            if neighbour not in atoms:
                if (
                    neighbour.atomic_number
                    > terminal_atoms.get(atom, off.Atom(0, 0, False)).atomic_number
                ):
                    terminal_atoms[atom] = neighbour
    # build out the torsion
    return tuple(
        [
            terminal_atoms[atoms[0]].molecule_atom_index,
            atoms[0].molecule_atom_index,
            atoms[1].molecule_atom_index,
            terminal_atoms[atoms[1]].molecule_atom_index,
        ]
    )


def get_symmetry_classes(molecule: off.Molecule) -> List[int]:
    """Calculate the symmetry classes of each atom in the molecule using the backend toolkits."""

    try:
        from rdkit import Chem

        rd_mol = molecule.to_rdkit()
        symmetry_classes = list(Chem.CanonicalRankAtoms(rd_mol, breakTies=False))

    except (ImportError, ModuleNotFoundError):
        from openeye import oechem

        oe_mol = molecule.to_openeye()
        oechem.OEPerceiveSymmetry(oe_mol)

        symmetry_classes_by_index = {
            a.GetIdx(): a.GetSymmetryClass() for a in oe_mol.GetAtoms()
        }
        symmetry_classes = [
            symmetry_classes_by_index[i] for i in range(molecule.n_atoms)
        ]

    return symmetry_classes


def get_symmetry_group(
    atom_group: Tuple[int, ...], symmetry_classes: List[int]
) -> Tuple[int, ...]:
    """
    For the list of atom groups calculate their symmetry class for the given molecule.
    """
    return tuple([symmetry_classes[atom] for atom in atom_group])
