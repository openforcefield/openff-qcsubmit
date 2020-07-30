from typing import Dict, List

from openforcefield import topology as off


def get_data(relative_path):
    """
    Get the file path to some data in the qcsubmit package.

    Parameters:
        relative_path: The relative path to the data
    """

    import os

    from pkg_resources import resource_filename

    fn = resource_filename("qcsubmit", os.path.join("data", relative_path))

    if not os.path.exists(fn):
        raise ValueError(
            f"Sorry! {fn} does not exist. If you just added it, you'll have to re-install"
        )

    return fn


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
