import copy
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

try:
    from openmm import unit
except ImportError:
    from simtk import unit

import networkx as nx
import numpy as np
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from tqdm import tqdm


def smirnoff_coverage(
    molecules: Iterable[Molecule], force_field: ForceField, verbose: bool = False
) -> Dict[str, Dict[str, int]]:
    """Returns a summary of how many of the specified molecules would be assigned each
    of the parameters in a force field.

    Notes:
        * Parameters which would not be assigned to any molecules of the specified
          molecules will not be included in the returned summary.

    Args:
        molecules: The molecules to generate a coverage report for.
        force_field: The force field containing the parameters to summarize.
        verbose: If true a progress bar will be shown on screen.

    Returns:
        A dictionary of the form ``coverage[handler_name][parameter_smirks] = count``
        which stores the number of molecules that would be assigned to each parameter.
    """

    molecules = [*molecules]

    coverage = defaultdict(lambda: defaultdict(set))

    for molecule in tqdm(
        molecules,
        total=len(molecules),
        ncols=80,
        disable=not verbose,
    ):

        full_labels = force_field.label_molecules(molecule.to_topology())[0]

        for handler_name, parameter_labels in full_labels.items():
            for parameter in parameter_labels.values():
                coverage[handler_name][parameter.smirks].add(
                    molecule.to_smiles(mapped=False, isomeric=False)
                )

    # Convert the defaultdict objects back into ordinary dicts so that users get
    # KeyError exceptions when trying to access missing handlers / smirks.
    return {
        handler_name: {smirks: len(count) for smirks, count in counts.items()}
        for handler_name, counts in coverage.items()
    }


def smirnoff_torsion_coverage(
    molecules: Iterable[Tuple[Molecule, Tuple[int, int, int, int]]],
    force_field: ForceField,
    verbose: bool = False,
):
    """Returns a summary of how many unique molecules within this collection
    would be assigned each of the parameters in a force field.

    Notes:
        * Parameters which would not be assigned to any molecules in the collection
          will not be included in the returned summary.

    Args:
        molecules: The molecules and associated torsion (as defined by a quartet of
            atom indices) to generate a coverage report for.
        force_field: The force field containing the parameters to summarize.
        verbose: If true a progress bar will be shown on screen.

    Returns:
        A dictionary of the form ``coverage[handler_name][parameter_smirks] = count``
        which stores the number of unique torsions within this collection that
        would be assigned to each parameter.
    """

    molecules = [*molecules]

    labelled_molecules = {}

    # Only label each unique molecule once as this is a pretty slow operation.
    for molecule, _ in tqdm(
        molecules,
        total=len(molecules),
        ncols=80,
        desc="Assigning Parameters",
        disable=not verbose,
    ):

        smiles = molecule.to_smiles(isomeric=False, mapped=False)

        if smiles in labelled_molecules:
            continue

        labelled_molecules[smiles] = force_field.label_molecules(
            molecule.to_topology()
        )[0]

    coverage = defaultdict(lambda: defaultdict(set))

    for molecule, dihedral in tqdm(
        molecules,
        total=len(molecules),
        ncols=80,
        desc="Summarising",
        disable=not verbose,
    ):

        smiles = molecule.to_smiles(isomeric=False, mapped=False)
        full_labels = labelled_molecules[smiles]

        tagged_molecule = copy.deepcopy(molecule)
        tagged_molecule.properties["atom_map"] = {
            j: i + 1 for i, j in enumerate(dihedral)
        }
        tagged_smiles = tagged_molecule.to_smiles(isomeric=False, mapped=True)

        dihedral_indices = {*dihedral[1:3]}

        for handler_name, parameter_labels in full_labels.items():
            for indices, parameter in parameter_labels.items():

                if handler_name not in {
                    "Bonds",
                    "Angles",
                    "ProperTorsions",
                    "ImproperTorsions",
                }:
                    continue

                consecutive_pairs = [{*pair} for pair in zip(indices, indices[1:])]

                # Only count angles and bonds involving the central dihedral bond or
                # dihedrals involving the central dihedral bond.
                if (
                    handler_name in {"Bonds", "Angles", "ImproperTorsions"}
                    and all(pair != dihedral_indices for pair in consecutive_pairs)
                ) or (
                    handler_name == "ProperTorsions"
                    and consecutive_pairs[1] != dihedral_indices
                ):
                    continue

                coverage[handler_name][parameter.smirks].add(tagged_smiles)

    return {
        handler_name: {smirks: len(smiles) for smirks, smiles in smiles.items()}
        for handler_name, smiles in coverage.items()
    }


def split_openff_molecule(molecule: Molecule) -> List[Molecule]:
    """
    For a gievn openff molecule split it into its component parts if it is actually a multi-component system.

    Args:
        molecule:
            The openff.toolkit.topology.Molecule which should be split.
    """
    sub_graphs = list(nx.connected_components(molecule.to_networkx()))
    if len(sub_graphs) == 1:
        return [
            molecule,
        ]
    component_molecules = []
    for sub_graph in sub_graphs:
        # map the old index to the new one
        index_mapping = {}
        comp_mol = Molecule()
        for atom in sub_graph:
            new_index = comp_mol.add_atom(**molecule.atoms[atom].to_dict())
            index_mapping[atom] = new_index
        for bond in molecule.bonds:
            if bond.atom1_index in sub_graph and bond.atom2_index in sub_graph:
                bond_data = {
                    "atom1": comp_mol.atoms[index_mapping[bond.atom1_index]],
                    "atom2": comp_mol.atoms[index_mapping[bond.atom2_index]],
                    "bond_order": bond.bond_order,
                    "stereochemistry": bond.stereochemistry,
                    "is_aromatic": bond.is_aromatic,
                    "fractional_bond_order": bond.fractional_bond_order,
                }
                comp_mol.add_bond(**bond_data)
        # move the conformers
        if molecule.n_conformers != 0:
            for conformer in molecule.conformers:
                new_conformer = np.zeros((comp_mol.n_atoms, 3))
                for i in sub_graph:
                    new_conformer[index_mapping[i]] = conformer[i]
                comp_mol.add_conformer(new_conformer * unit.angstrom)
        component_molecules.append(comp_mol)
    return component_molecules
