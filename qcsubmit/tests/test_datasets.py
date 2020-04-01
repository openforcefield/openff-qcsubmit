"""
Unit test for the vairous dataset classes in the package.
"""
import pytest
from ..datasets import ComponentResult, BasicDataSet
from openforcefield.topology import Molecule
from functools import lru_cache
import numpy as np
from simtk import unit


@lru_cache()
def duplicated_molecules(include_conformers: bool = True, duplicates: int = 2):
    """
    Return a list of duplicated molecules.

    Parameters:
        include_conformers: If the molecules should have conformers or not.
        duplicates: The number of times each molecule should be duplicated.
    """

    smiles = ['CCC', 'CCO', 'CCCC', 'c1ccccc1']

    molecules = []
    for smile in smiles:
        for i in range(duplicates):
            mol = Molecule.from_smiles(smile)
            if include_conformers:
                mol.generate_conformers()
            molecules.append(mol)

    return molecules


def test_componetresult_deduplication_standard():
    """
    Test the components results ability to deduplicate molecules.
    """

    result = ComponentResult(component_name='Test deduplication',
                             component_description={})

    assert result.component_name == 'Test deduplication'

    # test deduplication with no conformers
    molecules = duplicated_molecules(include_conformers=False, duplicates=2)

    for molecule in molecules:
        result.add_molecule(molecule)

    # make sure only 3 molecules were added to the component
    assert len(result.molecules) == 3
    assert len(result.filtered) == 0


def test_componentresult_deduplication_coordinates():
    """
    Test the component results ability to deduplicate molecules with coordinates.
    The conformers on the duplicated molecules should be the same and will not be transfered.
    """

    result = ComponentResult(component_name='Test deduplication',
                             component_description={})

    # test using conformers, conformers that are the same will be condensed
    molecules = duplicated_molecules(include_conformers=True, duplicates=2)

    for molecule in molecules:
        result.add_molecule(molecule)

    assert len(result.molecules) == 3
    for molecule in result.molecules:
        assert molecule.n_conformers == 1

    assert result.filtered == []


@pytest.mark.parametrize('duplicates',
                         [
                             pytest.param(2, id='two duplicates'),
                             pytest.param(4, id='four duplicates'),
                             pytest.param(6, id='six duplicates')

                         ])
def test_componentresult_deduplication_diff_coords(duplicates):
    """
    Test the componentresults ability to deduplicate molecules with different coordinates and condense them on to the
    same molecule.
    """

    result = ComponentResult(component_name='Test deduplication',
                             component_description={})

    # test using conformers that are different
    molecules = duplicated_molecules(include_conformers=False, duplicates=duplicates)

    # make some random coordinates
    for molecule in molecules:
        new_conformer = np.random.rand(molecule.n_atoms, 3)
        molecule.add_conformer(new_conformer * unit.angstrom)
        result.add_molecule(molecule)

    for molecule in result.molecules:
        assert molecule.n_conformers == duplicates
        for i in range(molecule.n_conformers):
            for j in range(molecule.n_conformers):
                if i != j:
                    assert molecule.conformers[i].tolist() != molecule.conformers[j].tolist()


def test_componentresult_deduplication_torsions():
    """
    Make sure that any torsion index results are correctly transfered when deduplicating molecules.
    """

    result = ComponentResult(component_name='Test deduplication',
                             component_description={})

    molecules = duplicated_molecules(include_conformers=False, duplicates=2)

    for molecule in molecules:
        molecule.properties['torsion_index'] = [np.random.randint(low=0, high=7, size=4).tolist()]

        result.add_molecule(molecule)

    for molecule in result.molecules:
        assert 'torsion_index' in molecule.properties
        assert len(molecule.properties['torsion_index']) == 2


def test_componentresult_filter_molecules():
    """
    Test component results ability to filter out molecules.
    """

    result = ComponentResult(component_name='Test filtering',
                             component_description={'component_name': 'TestFiltering',
                                                    'component_description': 'TestFiltering',
                                                    'component_fail_message': 'TestFiltering'})

    molecules = duplicated_molecules(include_conformers=True, duplicates=1)

    for molecule in molecules:
        result.add_molecule(molecule)

    assert len(result.molecules) == len(molecules)
    assert result.filtered == []

    for molecule in molecules:
        result.filter_molecule(molecule)

    # make sure all of the molecules have been removed and filtered
    assert result.molecules == []
    assert len(result.filtered) == len(molecules)

