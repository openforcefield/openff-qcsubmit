import copy
from collections import defaultdict
from typing import Dict, Iterable, Tuple

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
