"""
Unit test for the vairous dataset classes in the package.
"""

from typing import Dict, Tuple

import numpy as np
import pytest
from openforcefield.topology import Molecule
from pydantic import ValidationError
from simtk import unit

from qcsubmit.common_structures import TorsionIndexer
from qcsubmit.constraints import Constraints, PositionConstraintSet
from qcsubmit.datasets import (
    BasicDataset,
    ComponentResult,
    DatasetEntry,
    OptimizationDataset,
    TorsiondriveDataset,
)
from qcsubmit.exceptions import (
    AngleConnectionError,
    BondConnectionError,
    ConstraintError,
    DatasetCombinationError,
    DatasetInputError,
    DihedralConnectionError,
    LinearTorsionError,
    MissingBasisCoverageError,
    QCSpecificationError,
    UnsupportedFiletypeError,
)
from qcsubmit.factories import BasicDatasetFactory
from qcsubmit.testing import temp_directory
from qcsubmit.utils import (
    condense_molecules,
    get_data,
    update_specification_and_metadata,
)
from qcsubmit.validators import (
    check_angle_connection,
    check_bond_connection,
    check_torsion_connection,
)


def duplicated_molecules(include_conformers: bool = True, duplicates: int = 2):
    """
    Return a list of duplicated molecules.

    Parameters:
        include_conformers: If the molecules should have conformers or not.
        duplicates: The number of times each molecule should be duplicated.
    """

    smiles = ["CCC", "CCO", "CCCC", "c1ccccc1"]

    molecules = []
    for smile in smiles:
        for i in range(duplicates):
            mol = Molecule.from_smiles(smile)
            if include_conformers:
                mol.generate_conformers()
            molecules.append(mol)

    return molecules


def get_dihedral(molecule: Molecule) -> Tuple[int, int, int, int]:
    """
    Get a valid dihedral for the molecule.
    """
    torsion = list(molecule.propers)[0]
    dihedral = [atom.molecule_atom_index for atom in torsion]
    return tuple(dihedral)


def get_cmiles(molecule: Molecule) -> Dict[str, str]:
    """
    Generate a valid and full cmiles for the given molecule.
    """

    factory = BasicDatasetFactory()
    return factory.create_cmiles_metadata(molecule)


def test_componentresult_repr():
    """
    Make sure the __repr__ is working.
    """
    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={},
                             molecules=duplicated_molecules(include_conformers=False, duplicates=1))

    assert repr(result) == "ComponentResult(name=Test deduplication, molecules=4, filtered=0)"


def test_componentresult_str():
    """
    Make sure the __str__ is working.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={},
                             molecules=duplicated_molecules(include_conformers=False, duplicates=1))

    assert str(result) == "<ComponentResult name='Test deduplication' molecules='4' filtered='0'>"


def test_componetresult_deduplication_standard():
    """
    Test the components results ability to deduplicate molecules.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})

    assert result.component_name == "Test deduplication"

    # test deduplication with no conformers
    duplicates = 2
    molecules = duplicated_molecules(include_conformers=False, duplicates=duplicates)

    for molecule in molecules:
        result.add_molecule(molecule)

    # make sure only 1 copy of each molecule is added
    assert len(result.molecules) == len(molecules) / duplicates
    assert len(result.filtered) == 0


def test_componetresult_directory():
    """
    Test loading up some molecules from a driectory of files.
    """
    butane_conformers = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")
    butane = condense_molecules(butane_conformers)
    ethanol = Molecule.from_file(get_data("ethanol.sdf"), "sdf")
    with temp_directory():
        # write the files out
        ethanol.to_file("ethanol.sdf", "sdf")
        butane.to_file("butane.pdb", "pdb")

        # now make a result to load these molecules
        result = ComponentResult(component_name="test",
                                 component_description={},
                                 component_provenance={},
                                 input_directory=".")
        assert result.n_molecules == 2


@pytest.mark.parametrize("file_name", [
    pytest.param("benzene.sdf", id="SDF file"),
    pytest.param("butane_conformers.pdb", id="PDB file"),
    pytest.param("tautomers_small.smi", id="SMI file")
])
def test_componetresult_input_file(file_name):
    """
    Test loading up some molecules from an input file
    """
    result = ComponentResult(component_name="Test",
                             component_description={},
                             component_provenance={},
                             input_file=get_data(file_name))
    assert result.n_molecules > 0


def test_componentresult_deduplication_coordinates():
    """
    Test the component results ability to deduplicate molecules with coordinates.
    The conformers on the duplicated molecules should be the same and will not be transfered.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})

    # test using conformers, conformers that are the same will be condensed
    duplicates = 2
    molecules = duplicated_molecules(include_conformers=True, duplicates=duplicates)

    for molecule in molecules:
        result.add_molecule(molecule)

    assert len(result.molecules) == len(molecules) / duplicates
    for molecule in result.molecules:
        assert molecule.n_conformers == 1

    assert result.filtered == []


@pytest.mark.parametrize(
    "duplicates",
    [pytest.param(2, id="two duplicates"), pytest.param(4, id="four duplicates"), pytest.param(6, id="six duplicates")],
)
def test_componentresult_deduplication_diff_coords(duplicates):
    """
    Test the componentresults ability to deduplicate molecules with different coordinates and condense them on to the
    same molecule.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})

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


def test_componentresult_remove_molecule():
    """
    Test removing a molecule not in a dataset.
    """
    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})
    methanol = Molecule.from_file(get_data("methanol.sdf"), "sdf")
    result.filter_molecule(molecule=methanol)

    assert methanol in result.filtered


def test_componentresult_deduplication_torsions_same_bond_same_coords():
    """
    Make sure that the same rotatable bond is not highlighted more than once when deduplicating molecules.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})

    molecules = [Molecule.from_file(get_data("methanol.sdf"), 'sdf')] * 3
    ethanol_dihedrals = [(5, 1, 0, 2), (5, 1, 0, 3), (5, 1, 0, 4)]
    for molecule, dihedral in zip(molecules, ethanol_dihedrals):
        torsion_indexer = TorsionIndexer()
        torsion_indexer.add_torsion(torsion=dihedral, scan_range=None)
        molecule.properties["dihedrals"] = torsion_indexer
        result.add_molecule(molecule)

    # now make sure only one dihedral is selected
    assert len(result.molecules) == 1
    assert result.molecules[0].properties["dihedrals"].n_torsions == 1
    # this checks the bond has been ordered
    assert (0, 1) in result.molecules[0].properties["dihedrals"].torsions


@pytest.mark.parametrize("ethanol_data", [
    pytest.param(("ethanol.sdf",  [(8, 2, 1, 0)], None), id="correct torsion ethanol"),
    pytest.param(("ethanol.sdf",  [(22, 23, 24, 100)], DihedralConnectionError),
                 id="incorrect torsion ethanol"),
    pytest.param(("ethanol.sdf", [(7, 2, 1, 4)], DihedralConnectionError),
                 id="incorrect torsion ethanol"),
    pytest.param(("ethanol.sdf", [(8, 2, 1, 0), (4, 0, 1, 2)], None),
                 id="correct double torsion ethanol"),
    pytest.param(("ethanol.sdf", [(7, 2, 1, 4), (4, 0, 1, 2)],
                  DihedralConnectionError), id="incorrect double torsion ethanol"),
    pytest.param(("ethanol.sdf", [(3, 0, 4, 5)], None),
                 id="correct improper ethanol"),
    pytest.param(("ethanol.sdf", [(100, 0, 4, 5)], DihedralConnectionError),
                 id="incorrect improper ethanol index error"),
    pytest.param(("ethanol.sdf", [(7, 0, 4, 5)], DihedralConnectionError),
                 id="incorrect improper ethanol"),
    pytest.param(("benzene.sdf", [(0, 1, 2, 7)], None),
                 id="correct improper benzene"),
    pytest.param(("benzene.sdf", [(4, 0, 1, 2)], DihedralConnectionError),
                 id="incorrect improper benzene"),
])
def test_dataset_dihedral_validation(ethanol_data):
    """
    Test adding molecules to a dataset with different dihedrals, this will make sure that the dataset validators work.
    """

    dataset = TorsiondriveDataset()
    molecule_file, dihedrals, error = ethanol_data
    ethanol = Molecule.from_file(get_data(molecule_file))
    attributes = get_cmiles(ethanol)
    index = "test1"
    if error is not None:
        with pytest.raises(error):
            dataset.add_molecule(index=index, molecule=ethanol, attributes=attributes, dihedrals=dihedrals)
    else:
        dataset.add_molecule(index=index, molecule=ethanol, attributes=attributes, dihedrals=dihedrals)
        assert dataset.n_molecules == 1


def test_dataset_valence_validator():
    """
    Make sure a warning about the valence of a molecule with a net charge is produced.
    """
    import warnings
    dataset = BasicDataset()
    charged_molecules = Molecule.from_file(get_data("charged_molecules.smi"))
    for molecule in charged_molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        with pytest.warns(UserWarning):
            dataset.add_molecule(index=index, molecule=molecule, attributes=attributes)


def test_molecular_complex_validator():
    """
    Make sure that molecular complexes are caught by the validator.
    """

    from qcsubmit.exceptions import MolecularComplexError
    with pytest.raises(MolecularComplexError):
        _ = BasicDataset.parse_file(get_data("molecular_complex.json"))


def test_dataset_linear_dihedral_validator():
    """
    Make sure that dataset rejects molecules with tagged linear bonds.
    """

    from qcsubmit.factories import TorsiondriveDatasetFactory
    dataset = BasicDataset()
    molecules = Molecule.from_file(get_data("linear_molecules.sdf"), allow_undefined_stereo=True)
    factory = TorsiondriveDatasetFactory()
    linear_smarts = "[*!D1:1]~[$(*#*)&D2,$(C=*)&D2:2]"

    # for each molecule we want to tag each linear dihedral
    for molecule in molecules:
        matches = molecule.chemical_environment_matches(linear_smarts)
        bond = molecule.get_bond_between(*matches[0])
        dihedral = factory._get_torsion_string(bond)
        attributes = get_cmiles(molecule)
        with pytest.raises(LinearTorsionError):
            dataset.add_molecule(index="linear test", molecule=molecule, attributes=attributes, dihedrals=[dihedral, ])


@pytest.mark.parametrize("constraint_settings", [
    pytest.param(("distance", [0, 1], None), id="distance correct order"),
    pytest.param(("angle", [0, 1, 2], None), id="angle correct order"),
    pytest.param(("dihedral", [0, 1, 2, 3], None), id="dihedral correct order"),
    pytest.param(("xyz", [0], None), id="position correct"),
    pytest.param(("xyz", [0, 1, 2, 3, 4, 5, 6], None), id="position many"),
    pytest.param(("xyz", [0, 0, 1, 2], ConstraintError), id="position not unique"),
    pytest.param(("distance", [0, 1, 2], ConstraintError), id="distance too many indices"),
    pytest.param(("bond", [0, 1], ConstraintError), id="invalid constraint type.")
])
def test_add_freeze_constraints_validator(constraint_settings):
    """
    Test out adding multiple types of constraints and their validators.
    """
    cons = Constraints()
    assert cons.has_constraints is False
    con_type, indices, error = constraint_settings
    if error is not None:
        with pytest.raises(error):
            cons.add_freeze_constraint(constraint_type=con_type, indices=indices)
    else:
        cons.add_freeze_constraint(constraint_type=con_type, indices=indices)
        # now make sure that the indices are being ordered by trying to add a constraint in reverse
        cons.add_freeze_constraint(constraint_type=con_type, indices=list(reversed(indices)))
        assert len(cons.freeze) == 1
        # make sure the class knows it know has constraints
        assert cons.has_constraints is True
        cons_dict = cons.dict()
        # make sure we drop empty constraint lists
        assert "set" not in cons_dict


@pytest.mark.parametrize("constraint_settings", [
    pytest.param(("distance", [0, 1], 1, None), id="distance correct order"),
    pytest.param(("angle", [0, 1, 2], 100, None), id="angle correct order"),
    pytest.param(("dihedral", [0, 1, 2, 3], 50, None), id="dihedral correct order"),
    pytest.param(("xyz", [0], [0, 1, 2], None), id="position correct"),
    pytest.param(("xyz", [0, 1, 2, 3, 4, 5, 6], [0, 1, 0], ConstraintError), id="position too many atoms"),
    pytest.param(("distance", [0, 1, 2], 1, ConstraintError), id="distance too many indices"),
    pytest.param(("bond", [0, 1], 1, ConstraintError), id="invalid constraint type.")
])
def test_add_set_constraints_validator(constraint_settings):
    """
    Test out adding multiple types of constraints and their validators.
    """
    cons = Constraints()
    assert cons.has_constraints is False
    con_type, indices, value, error = constraint_settings
    if error is not None:
        with pytest.raises(error):
            cons.add_set_constraint(constraint_type=con_type, indices=indices, value=value)
    else:
        cons.add_set_constraint(constraint_type=con_type, indices=indices, value=value)
        # now make sure that the indices are being ordered by trying to add a constraint in reverse
        cons.add_set_constraint(constraint_type=con_type, indices=list(reversed(indices)), value=value)
        assert len(cons.set) == 1
        # make sure the class knows it know has constraints
        assert cons.has_constraints is True
        cons_dict = cons.dict()
        # make sure we drop empty constraint lists
        assert "freeze" not in cons_dict


@pytest.mark.parametrize("constraint_settings", [
    pytest.param(("0, -1, -2", None), id="correct space list"),
    pytest.param(("0 0 0 ", None), id="correct space list"),
    pytest.param(("0-0-0", ConstraintError), id="incorrect hyphen list"),
    pytest.param(("C, a, b", ConstraintError), id="none float position"),
    pytest.param(("0,0,0", None), id="correct comma list"),
    pytest.param(([0, 0, 0], None), id="correct list")

])
def test_position_set_constraint_validation(constraint_settings):
    """
    Test each of the valid inputs for the position set constraint and make sure the value is converted correctly.
    """
    value, error = constraint_settings
    if error is None:
        constraint = PositionConstraintSet(indices=(0, ), value=value)
        # make sure a single string is made
        assert len(constraint.value.split()) == 3
    else:
        with pytest.raises(error):
            PositionConstraintSet(indices=(0, ), value=value)


@pytest.mark.parametrize("constraint_settings", [
    pytest.param(("freeze", "dihedral", [0, 1, 2, 3], None, ConstraintError), id="freeze dihedral valid"),
    pytest.param(("set", "dihedral", [0, 1, 2, 3], 50, ConstraintError), id="set dihedral valid"),
    pytest.param(("scan", "dihedral", [0, 1, 2, 3], 50, ConstraintError), id="invalid scan constraint"),
])
def test_add_constraints_to_entry_errors(constraint_settings):
    """
    Test adding new constraints to a valid dataset entry, make sure errors are raised if we say the constraint is bonded but it is not.
    """
    mol = Molecule.from_smiles("CC")
    entry = DatasetEntry(off_molecule=mol, attributes=get_cmiles(mol), index=mol.to_smiles(), keywords={}, extras={})
    # get the constraint info
    constraint, constraint_type, indices, value, error = constraint_settings
    with pytest.raises(error):
            entry.add_constraint(constraint=constraint, constraint_type=constraint_type, indices=indices, value=value)


@pytest.mark.parametrize("constraint_settings", [
    pytest.param(("freeze", "dihedral", [3, 0, 1, 5], None, True, None), id="freeze dihedral"),
    pytest.param(("set", "angle", [3, 0, 1], 50, True, None), id="set angle"),
    pytest.param(("freeze", "distance", [0, 1], None, True, None), id="freeze distance"),
    pytest.param(("scan", "dihedral", [0, 1, 2, 3], 50, True, ConstraintError), id="invalid scan constraint"),
])
def test_add_constraints_to_entry(constraint_settings):
    """
    Test adding new constraints to a valid dataset entry make sure no warnings are raised as the constraints are bonded.
    """
    mol = Molecule.from_smiles("CC")
    entry = DatasetEntry(off_molecule=mol, attributes=get_cmiles(mol), index=mol.to_smiles(), keywords={}, extras={})
    # get the constraint info
    constraint, constraint_type, indices, value, bonded, error = constraint_settings
    if error is None:
        with pytest.warns(None) as all_warnings:
            entry.add_constraint(constraint=constraint, constraint_type=constraint_type, indices=indices, bonded=bonded, value=value)
        # make sure no warnings were raised
        assert len(all_warnings) == 0
        assert entry.constraints.has_constraints is True
        assert "constraints" in entry.formatted_keywords

    else:
        with pytest.raises(error):
            entry.add_constraint(constraint=constraint, constraint_type=constraint_type, indices=indices, bonded=bonded, value=value)


@pytest.mark.parametrize("constraint_setting", [
    pytest.param("constraints", id="constraint"),
    pytest.param("keywords", id="keywords"),
    pytest.param(None, id="no constraints")
])
def test_add_entry_with_constraints(constraint_setting):
    """
    Add an entry to a dataset with constraints in the keywords and make sure they converted to the constraints field.
    """
    dataset = BasicDataset()
    # now add a molecule with constraints in the keywords
    mol = Molecule.from_file(get_data("ethane.sdf"), "sdf")
    constraints = {"set": [{"type": "dihedral", "indices": [0, 1, 2, 3], "value": 50, "bonded": False},
                           {"type": "angle", "indices": [0, 1, 2], "value": 50, "bonded": False},
                           {"type": "distance", "indices": [1, 2], "value": 1, "bonded": False}]}
    index = mol.to_smiles()

    if constraint_setting == "constraints":
        dataset.add_molecule(index=index, molecule=mol, attributes=get_cmiles(mol), constraints=constraints)
    elif constraint_setting == "keywords":
        dataset.add_molecule(index=index, molecule=mol, attributes=get_cmiles(mol), keywords={"constraints": constraints})
    elif constraint_setting is None:
        dataset.add_molecule(index=index, molecule=mol, attributes=get_cmiles(mol))

    entry = dataset.dataset[index]
    if constraint_setting is not None:
        assert "constraints" in entry.formatted_keywords
        assert entry.constraints.has_constraints is True
    else:
        assert entry.keywords == entry.formatted_keywords
        assert entry.constraints.has_constraints is False


@pytest.mark.parametrize("atom_data", [
    pytest.param(([1, 2], check_bond_connection, BondConnectionError), id="Bond connection error"),
    pytest.param(([0, 1, 2], check_angle_connection, AngleConnectionError), id="Angle connection error"),
    pytest.param(([5, 0, 1, 2], check_torsion_connection, DihedralConnectionError), id="Dihedral connection error")
])
def test_check_connection_types_error(atom_data):
    """
    Make sure the connection checks raise the correct errors.
    """
    ethane = Molecule.from_file(get_data("ethane.sdf"), "sdf")
    atoms, check, error = atom_data
    with pytest.raises(error):
        check(atoms, ethane)


def test_constraints_are_equall():
    """
    Test if two constraints are equal.
    """
    const1 = Constraints()
    const2 = Constraints()
    assert const1 == const2

    # now add a constraint
    const1.add_freeze_constraint(constraint_type="dihedral", indices=[0, 1, 2, 3])
    assert const1 != const2
    const1.add_set_constraint(constraint_type="angle", indices=[0, 1, 2], value=90)
    assert const1 != const2

    # now add more restraints in different orders
    const2.add_freeze_constraint(constraint_type="distance", indices=[0, 1])
    const2.add_freeze_constraint(constraint_type="dihedral", indices=[0, 1, 2, 3])
    const2.add_set_constraint(constraint_type="angle", indices=[0, 1, 2], value=90)

    const1.add_freeze_constraint(constraint_type="distance", indices=[0, 1])
    assert const1 == const2


def test_scf_prop_validation():
    """
    Make sure unsupported scf properties are not allowed into a dataset.
    """
    dataset = BasicDataset()
    with pytest.raises(DatasetInputError):
        dataset.scf_properties = ["ddec_charges"]


def test_add_molecule_no_extras():
    """
    Test that adding a molecule with no extras automatically updates the extras to include the cmiles.
    """
    dataset = BasicDataset()
    mols = duplicated_molecules(include_conformers=True, duplicates=1)
    for mol in mols:
        index = mol.to_smiles()
        attributes = get_cmiles(mol)
        dataset.add_molecule(index=index, molecule=mol, attributes=attributes)

    for entry in dataset.dataset.values():
        for mol in entry.initial_molecules:
            assert "canonical_isomeric_explicit_hydrogen_mapped_smiles" in mol.extras


@pytest.mark.parametrize("dataset_data", [
    pytest.param((BasicDataset, "OpenFF NCI250K Boron 1", ["default"]), id="Dataset no metadata"),
    pytest.param((OptimizationDataset, "OpenFF NCI250K Boron 1", ["default"]), id="OptimizationDataset no metadata"),
    pytest.param((TorsiondriveDataset, "OpenFF Fragmenter Phenyl Benchmark", ["UFF", "B3LYP-D3"]), id="TorsiondriveDataset no metadata"),
    pytest.param((TorsiondriveDataset, "OpenFF Rowley Biaryl v1.0", ["default"]), id="Torsiondrive with metadata")
])
def test_dataset_update(dataset_data):
    """
    Make sure the utils function can update elements and pull the correct specs.
    """
    import qcportal as ptl
    dataset_type, dataset_name, specs = dataset_data
    client = ptl.FractalClient()
    # set up the dataset
    dataset = dataset_type(dataset_name=dataset_name)
    assert bool(dataset.metadata.elements) is False
    dataset = update_specification_and_metadata(dataset, client)
    # now make sure the elements have been updated and the spec added
    assert bool(dataset.metadata.elements) is True
    for spec in specs:
        assert spec in dataset.qc_specifications


@pytest.mark.parametrize("dataset_types", [
    pytest.param((BasicDataset, OptimizationDataset), id="basic + optimization error"),
    pytest.param((OptimizationDataset, TorsiondriveDataset), id="optimization + torsiondrive error"),
    pytest.param((TorsiondriveDataset, BasicDataset), id="torsiondrive + basic error"),

])
def test_dataset_adding(dataset_types):
    """
    Test combining two different datasets this should raise an error as they have different conditions to be considered equal.
    """
    data_type1, data_type2 = dataset_types
    dataset1, dataset2 = data_type1(), data_type2()
    with pytest.raises(DatasetCombinationError):
        _ = dataset1 + dataset2


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"),
    pytest.param(OptimizationDataset, id="Optimizationdataset"),
])
def test_basic_and_opt_dataset_addition(dataset_type):
    """
    Test adding two basic and opt datasets together.
    Opt has no constraints and behaves like basic in this case.
    """

    dataset1 = dataset_type()
    dataset2 = dataset_type()
    # get some butane conformers and split them between the datasets
    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), 'pdb')
    butane1 = condense_molecules(molecules[:4])
    butane2 = condense_molecules(molecules[4:])
    dataset1.add_molecule(index=butane1.to_smiles(),
                          molecule=butane1,
                          attributes=get_cmiles(butane1))
    dataset2.add_molecule(index=butane2.to_smiles(),
                          molecule=butane2,
                          attributes=get_cmiles(butane2))
    # now check the number of molecules in each dataset
    assert dataset1.n_molecules == 1
    assert dataset1.n_records == 4
    assert dataset2.n_molecules == 1
    assert dataset2.n_records == 3

    # now add them
    new_dataset = dataset1 + dataset2
    assert new_dataset.n_molecules == 1
    assert new_dataset.n_records == 7

    # now add another molecule to dataset2
    ethanol = Molecule.from_file(get_data("ethanol.sdf"), "sdf")
    dataset2.add_molecule(index=ethanol.to_smiles(),
                          molecule=ethanol,
                          attributes=get_cmiles(ethanol))

    new_dataset = dataset1 + dataset2
    assert "O" in new_dataset.metadata.elements
    assert new_dataset.n_molecules == 2
    assert new_dataset.n_records == 8


def test_optimization_dataset_addition_constraints_equal():
    """
    Test adding two optimizationdatasets with constraints.
    """
    dataset1 = OptimizationDataset()
    dataset2 = OptimizationDataset()
    # add molecules to each dataset
    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), 'pdb')
    butane1 = condense_molecules(molecules[:4])
    butane2 = condense_molecules(molecules[4:])
    constraints = Constraints()
    constraints.add_set_constraint(constraint_type="dihedral", indices=[0, 1, 2, 3], value=60)
    constraints.add_freeze_constraint(constraint_type="distance", indices=[0, 1])
    dataset1.add_molecule(index=butane1.to_smiles(),
                          molecule=butane1,
                          attributes=get_cmiles(butane1),
                          constraints=constraints)
    dataset2.add_molecule(index=butane2.to_smiles(),
                          molecule=butane2,
                          attributes=get_cmiles(butane2),
                          constraints=constraints)
    # make sure the records are added
    assert dataset1.n_molecules == 1
    assert dataset1.n_records == 4

    assert dataset2.n_molecules == 1
    assert dataset2.n_records == 3

    # make sure the constraints are present
    for entry in dataset1.dataset.values():
        assert entry.constraints.has_constraints is True
    for entry in dataset2.dataset.values():
        assert entry.constraints.has_constraints is True

    new_dataset = dataset1 + dataset2
    # make the constraints are considered the same and the geometries are transferred
    assert new_dataset.n_molecules == 1
    assert new_dataset.n_records == 7


def test_optimization_dataset_addition_constraints_not_equall():
    """
    Test adding two optimizationdatasets with constraints that are not the same.
    """
    dataset1 = OptimizationDataset()
    dataset2 = OptimizationDataset()
    # add molecules to each dataset
    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), 'pdb')
    butane1 = condense_molecules(molecules[:4])
    butane2 = condense_molecules(molecules[4:])
    # make different constraints
    constraints1 = Constraints()
    constraints1.add_freeze_constraint(constraint_type="angle", indices=[0, 1, 2])
    constraints2 = Constraints()
    constraints2.add_set_constraint(constraint_type="distance", indices=[0, 1], value=1.5)

    dataset1.add_molecule(index=butane1.to_smiles(),
                          molecule=butane1,
                          attributes=get_cmiles(butane1),
                          constraints=constraints1)
    dataset2.add_molecule(index=butane2.to_smiles(),
                          molecule=butane2,
                          attributes=get_cmiles(butane2),
                          constraints=constraints2)
    # make sure the records are added
    assert dataset1.n_molecules == 1
    assert dataset1.n_records == 4

    assert dataset2.n_molecules == 1
    assert dataset2.n_records == 3

    # make sure the constraints are present
    for entry in dataset1.dataset.values():
        assert entry.constraints.has_constraints is True
    for entry in dataset2.dataset.values():
        assert entry.constraints.has_constraints is True

    # now add the datasets
    new_dataset = dataset1 + dataset2
    # there is still one unique molecule
    assert new_dataset.n_molecules == 1
    assert new_dataset.n_records == 7
    # there should be two entries however
    assert len(new_dataset.dataset) == 2


def test_torsiondrive_dataset_addition_same_dihedral():
    """
    Test adding together two different torsiondrive datasets.
    """
    from qcsubmit.factories import TorsiondriveDatasetFactory
    factory = TorsiondriveDatasetFactory()
    dataset1 = TorsiondriveDataset()
    dataset2 = TorsiondriveDataset()
    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), 'pdb')
    butane1 = condense_molecules(molecules[:4])
    butane2 = condense_molecules(molecules[4:])
    # select the same dihedral in both molecules
    dihedral = [(0, 1, 2, 3)]
    atom_map = {0: 0, 1: 1, 2: 2, 3: 3}
    butane1.properties["atom_map"] = atom_map
    dataset1.add_molecule(index=factory.create_index(butane1),
                          molecule=butane1,
                          attributes=get_cmiles(butane2),
                          dihedrals=dihedral)
    dataset2.add_molecule(index=factory.create_index(butane1),
                          molecule=butane2,
                          attributes=get_cmiles(butane2),
                          dihedrals=dihedral)
    assert dataset1.n_molecules == 1
    assert dataset1.n_records == 1
    assert dataset2.n_molecules == 1
    assert dataset2.n_records == 1

    new_dataset = dataset1 + dataset2

    assert new_dataset.n_molecules == 1
    assert new_dataset.n_records == 1
    for entry in new_dataset.dataset.values():
        assert len(entry.initial_molecules) == 7


def test_torsiondrive_dataset_addition_different_dihedral():
    """
    Test adding together two different torsiondrive datasets.
    """
    from qcsubmit.factories import TorsiondriveDatasetFactory
    factory = TorsiondriveDatasetFactory()
    dataset1 = TorsiondriveDataset()
    dataset2 = TorsiondriveDataset()
    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), 'pdb')
    butane1 = condense_molecules(molecules[:4])
    butane2 = condense_molecules(molecules[4:])
    # select different dihedrals in the molecules
    dihedral1 = [(0, 1, 2, 3)]
    atom_map1 = {0: 0, 1: 1, 2: 2, 3: 3}
    dihedral2 = [(4, 0, 1, 2)]
    atom_map2 = {4: 0, 0: 1, 1: 2, 2: 3}
    # set up the index
    butane1.properties["atom_map"] = atom_map1
    index1 = factory.create_index(butane1)
    del butane1.properties["atom_map"]

    butane2.properties["atom_map"] = atom_map2
    index2 = factory.create_index(butane2)
    del butane2.properties["atom_map"]


    dataset1.add_molecule(index=index1,
                          molecule=butane1,
                          attributes=get_cmiles(butane1),
                          dihedrals=dihedral1)
    dataset2.add_molecule(index=index2,
                          molecule=butane2,
                          attributes=get_cmiles(butane2),
                          dihedrals=dihedral2)
    assert dataset1.n_molecules == 1
    assert dataset1.n_records == 1
    assert dataset2.n_molecules == 1
    assert dataset2.n_records == 1

    new_dataset = dataset1 + dataset2

    assert new_dataset.n_molecules == 1
    assert new_dataset.n_records == 2


@pytest.mark.parametrize("dataset_data", [
    pytest.param(("valid_torsion_dataset.json", None), id="valid torsion dataset"),
    pytest.param(("invalid_torsion_dataset.json", DihedralConnectionError), id="invalid torsion dataset"),
    pytest.param(("valid_double_torsion_dataset.json", None), id="valid double torsion dataset"),
    pytest.param(("invalid_double_torsion_dataset.json", DihedralConnectionError), id="invalid double torsion dataset"),
    pytest.param(("valid_improper_dataset.json", None), id="valid improper dataset"),
    pytest.param(("invalid_improper_dataset.json", DihedralConnectionError), id="invalid improper dataset"),
    pytest.param(("invalid_linear_dataset.json", LinearTorsionError), id="invalid linear dataset"),
])
def test_importing_dihedral_dataset(dataset_data):
    """
    Make sure that the dataset validators run when importing datasets.
    """

    dataset_name, error = dataset_data
    if error is None:
        dataset = TorsiondriveDataset.parse_file(get_data(dataset_name))
        assert dataset.n_molecules == 1
    else:
        with pytest.raises(error):
            _ = TorsiondriveDataset.parse_file(get_data(dataset_name))


def test_componenetresult_deduplication_torsions_same_bond_different_coords():
    """
    Make sure that similar molecules with different coords but the same selected rotatable bonds are correctly handled.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})

    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), 'pdb')
    butane_dihedral = (0, 1, 2, 3)
    for molecule in molecules:
        torsion_indexer = TorsionIndexer()
        torsion_indexer.add_torsion(torsion=butane_dihedral, scan_range=None)
        molecule.properties["dihedrals"] = torsion_indexer
        result.add_molecule(molecule)

    assert len(result.molecules) == 1
    assert result.molecules[0].n_conformers == 7
    assert result.molecules[0].properties["dihedrals"].n_torsions == 1
    assert result.n_molecules == 1
    assert result.n_conformers == 7
    assert result.n_filtered == 0


def test_componentresult_deduplication_torsions_1d():
    """
    Make sure that any torsion index results are correctly transferred when deduplicating molecules.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})

    duplicates = 2
    molecules = duplicated_molecules(include_conformers=False, duplicates=duplicates)
    dihedrals = [((0, 1, 2, 3), (-165, 150)), ((1, 2, 3, 4), (0, 180))]

    for i, molecule in enumerate(molecules):
        torsion_indexer = TorsionIndexer()
        torsion_indexer.add_torsion(*dihedrals[i % 2])
        molecule.properties["dihedrals"] = torsion_indexer

        result.add_molecule(molecule)

    for molecule in result.molecules:
        assert "dihedrals" in molecule.properties
        assert molecule.properties["dihedrals"].n_torsions == duplicates


def test_componentresult_deduplication_torsions_2d():
    """
    Make sure that any torsion index results are correctly transferred when deduplicating molecules.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})

    duplicates = 2
    molecules = duplicated_molecules(include_conformers=False, duplicates=duplicates)[:2]
    dihedrals = [((0, 1, 2, 3), (-165, 150)), ((1, 2, 3, 4), (0, 180))]
    double_dihedrals = [((0, 1, 2, 3), (1, 2, 3, 4), (-165, 150), (0, 180)), ((1, 2, 3, 4), (5, 6, 7, 8), (125, 150), (-10, 180))]
    for i, molecule in enumerate(molecules):
        torsion_indexer = TorsionIndexer()
        torsion_indexer.add_torsion(*dihedrals[i % 2])

        torsion_indexer.add_double_torsion(*double_dihedrals[i % 2])

        molecule.properties["dihedrals"] = torsion_indexer

        result.add_molecule(molecule)

    for molecule in result.molecules:
        assert "dihedrals" in molecule.properties
        assert molecule.properties["dihedrals"].n_torsions == duplicates
        assert molecule.properties["dihedrals"].n_double_torsions == duplicates


def test_torsion_indexing_torsion():
    """
    Test the torsion indexer class.
    """

    torsion_indexer = TorsionIndexer()
    # add a 1-D torsion
    torsion_indexer.add_torsion((3, 2, 1, 0), (180, -165))
    # make sure they have been ordered
    assert (1, 2) in torsion_indexer.torsions
    single_torsion = torsion_indexer.torsions[(1, 2)]
    assert single_torsion.scan_range1 == (-165, 180)
    assert single_torsion.get_dihedrals == [(0, 1, 2, 3), ]
    assert single_torsion.get_scan_range == [(-165, 180), ]
    assert single_torsion.get_atom_map == {0: 0, 1: 1, 2: 2, 3: 3}

    assert torsion_indexer.n_torsions == 1
    # test overwrite
    torsion_indexer.add_torsion(torsion=(3, 2, 1, 0), scan_range=None, overwrite=True)
    assert torsion_indexer.n_torsions == 1
    single_torsion = torsion_indexer.torsions[(1, 2)]
    assert single_torsion.get_scan_range is None


def test_torsion_indexing_double():
    """
    Test the torsion indexer with double torsions.
    """

    torsion_indexer = TorsionIndexer()
    # add a 2-D scan
    torsion_indexer.add_double_torsion(torsion1=(9, 8, 7, 6), torsion2=(0, 1, 2, 3), scan_range1=[40, -40],
                                       scan_range2=[-165, 180])
    # check the central bond was ordered
    assert ((1, 2), (7, 8)) in torsion_indexer.double_torsions
    double_torsion = torsion_indexer.double_torsions[((1, 2), (7, 8))]
    assert double_torsion.scan_range1 == (-40, 40)
    assert double_torsion.scan_range2 == (-165, 180)
    assert double_torsion.get_dihedrals == [(6, 7, 8, 9), (0, 1, 2, 3)]
    assert double_torsion.get_scan_range == [(-40, 40), (-165, 180)]
    assert double_torsion.get_atom_map == {0: 4, 1: 5, 2: 6, 3: 7, 6: 0, 7: 1, 8: 2, 9: 3}
    assert torsion_indexer.n_double_torsions == 1

    # test overwrite
    torsion_indexer.add_double_torsion(torsion1=(9, 8, 7, 6), torsion2=(0, 1, 2, 3), scan_range1=None,
                                       scan_range2=None, overwrite=True)
    assert torsion_indexer.n_double_torsions == 1
    double_torsion = torsion_indexer.double_torsions[((1, 2), (7, 8))]
    assert double_torsion.get_scan_range is None


def test_torsion_indexing_improper():
    """
    Test the torsion indexer with improper torsions.
    """

    torsion_indexer = TorsionIndexer()
    torsion_indexer.add_improper(1, (0, 1, 2, 3), scan_range=[40, -40])
    assert 1 in torsion_indexer.imporpers
    assert torsion_indexer.n_impropers == 1
    improper = torsion_indexer.imporpers[1]
    assert improper.get_scan_range == [(-40, 40), ]
    torsion_indexer.add_improper(1, (3, 2, 1, 0), scan_range=None, overwrite=True)
    # make sure it was over writen
    assert 1 in torsion_indexer.imporpers
    assert torsion_indexer.n_impropers == 1
    improper = torsion_indexer.imporpers[1]
    assert improper.get_dihedrals == [(3, 2, 1, 0), ]
    assert improper.get_scan_range is None
    assert improper.get_atom_map == {3: 0, 2: 1, 1: 2, 0: 3}


def test_torsion_index_iterator():
    """
    Make sure the iterator combines all torsions together.
    """
    from qcsubmit.common_structures import DoubleTorsion, ImproperTorsion, SingleTorsion
    torsion_indexer = TorsionIndexer()
    torsion_indexer.add_torsion((3, 2, 1, 0), (180, -165))
    torsion_indexer.add_double_torsion(torsion1=(9, 8, 7, 6), torsion2=(0, 1, 2, 3), scan_range1=[40, -40],
                                       scan_range2=[-165, 180])
    torsion_indexer.add_improper(1, (0, 1, 2, 3), scan_range=[40, -40])
    assert torsion_indexer.n_torsions == 1
    assert torsion_indexer.n_double_torsions == 1
    assert torsion_indexer.n_impropers == 1
    dihedrals = torsion_indexer.get_dihedrals
    assert len(dihedrals) == 3
    assert isinstance(dihedrals[0], SingleTorsion)
    assert isinstance(dihedrals[1], DoubleTorsion)
    assert isinstance(dihedrals[2], ImproperTorsion)


def test_torsion_indexer_update_no_mapping():
    """
    Test updating one torsion indexer with another with no mapping.
    """

    torsion_indexer1 = TorsionIndexer()
    torsion_indexer1.add_torsion((0, 1, 2, 3))
    torsion_indexer1.add_double_torsion(torsion1=(0, 1, 2, 3), torsion2=(9, 8, 7, 6))
    torsion_indexer1.add_improper(1, (0, 1, 2, 3))
    assert torsion_indexer1.n_torsions == 1
    assert torsion_indexer1.n_double_torsions == 1
    assert torsion_indexer1.n_impropers == 1

    torsion_indexer2 = TorsionIndexer()
    torsion_indexer2.add_torsion((9, 8, 7, 6))
    torsion_indexer2.add_double_torsion(torsion1=(9, 8, 7, 6), torsion2=(10, 11, 12, 13))
    torsion_indexer2.add_improper(5, (5, 6, 7, 8))
    assert torsion_indexer2.n_torsions == 1
    assert torsion_indexer2.n_double_torsions == 1
    assert torsion_indexer2.n_impropers == 1

    # update 1 with 2
    torsion_indexer1.update(torsion_indexer2)
    assert torsion_indexer1.n_torsions == 2
    assert torsion_indexer1.n_double_torsions == 2
    assert torsion_indexer1.n_impropers == 2


def test_componentresult_filter_molecules():
    """
    Test component results ability to filter out molecules.
    """

    result = ComponentResult(
        component_name="Test filtering",
        component_description={
            "component_name": "TestFiltering",
            "component_description": "TestFiltering",
            "component_fail_message": "TestFiltering",
        },
        component_provenance={},
    )

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


def test_dataset_add_filter_molecules():
    """
    Test adding extra molecules to filter.
    """
    dataset = BasicDataset()
    methane = Molecule.from_smiles("C")
    ethanol = Molecule.from_smiles("CC")
    dataset.filter_molecules(molecules=methane,
                             component_name="test",
                             component_description={},
                             component_provenance={})
    # now we want to add this extra molecule to this group
    dataset.filter_molecules(molecules=ethanol,
                             component_name="test",
                             component_description={},
                             component_provenance={})
    filtered = dataset.filtered_molecules["test"]
    assert len(filtered.molecules) == 2


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"), pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_dataset_metadata(dataset_type):
    """
    Test that the metadata for each dataset type s correctly assigned.
    """

    # make a basic dataset
    dataset = dataset_type(dataset_name="Testing dataset name",
                           dataset_tagline="test tagline",
                           description="Test description")

    # check the metadata
    empty_fields = dataset.metadata.validate_metadata(raise_errors=False)
    # this should be the only none autofilled field
    assert empty_fields == ["long_description_url"]

    # now make sure the names and types match
    assert dataset.metadata.dataset_name == dataset.dataset_name
    assert dataset.metadata.short_description == dataset.dataset_tagline
    assert dataset.metadata.long_description == dataset.description
    assert dataset.metadata.collection_type == dataset.dataset_type


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"), pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_wrong_metadata_collection_type(dataset_type):
    """
    Test passing in the wrong collection type into the metadata this should be corrected during the init.
    """

    from qcsubmit.common_structures import Metadata
    meta = Metadata(collection_type="INVALID")
    dataset = dataset_type(metadata=meta)

    # make sure the init of the dataset corrects the collection type
    assert dataset.metadata.collection_type != "INVALID"
    assert dataset.metadata.collection_type == dataset.dataset_type


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"),
    pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_dataset_exporting_same_type(dataset_type):
    """
    Test making the given dataset from the json of another instance of the same dataset type.
    """

    with temp_directory():
        dataset = dataset_type(compute_tag="test tag")
        dataset.export_dataset('dataset.json')

        dataset2 = dataset_type.parse_file('dataset.json')
        assert dataset2.compute_tag == "test tag"
        assert dataset.metadata == dataset2.metadata


def test_dataset_unsupported_exports():
    """
    Test trying to export to unsupported file types.
    """
    with temp_directory():
        dataset = BasicDataset()
        with pytest.raises(UnsupportedFiletypeError):
            dataset.export_dataset("dataset.yaml")


@pytest.mark.parametrize("molecule_data", [
    pytest.param((Molecule.from_smiles("CC"), 0), id="Molecule 0 entries"),
    pytest.param(("CC", 0), id="Smiles 0 entries"),
    pytest.param(("CCC", 1), id="Smiles 1 entries"),
])
def test_get_molecule_entry(molecule_data):
    """
    Test getting a molecule entry from dataset.
    """
    dataset = BasicDataset()
    query_molecule, entries_no = molecule_data
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)

    # check for molecule entries
    entries = dataset.get_molecule_entry(query_molecule)
    assert len(entries) == entries_no


def test_get_entry_molecule():
    """
    Test getting a molecule with and without conformers from a dataset.
    """
    dataset = BasicDataset()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    for molecule in molecules:
        index =molecule.to_smiles()
        attributes = get_cmiles(molecule)
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)

    # check the conformers are attached when requested
    for entry in dataset.dataset.values():
        mol_no_con = entry.get_off_molecule(include_conformers=False)
        mol_con = entry.get_off_molecule(include_conformers=True)
        assert mol_no_con.n_conformers == 0
        assert mol_con.n_conformers == 1
        assert mol_no_con == mol_con


def test_dataset_nmolecules_tautomers():
    """
    We use inchikey to find the unique molecules however some tautomers can have the same inchikey so make sure we can still find the unique molecules by forceing the inchi_key to be the same.
    """

    dataset = BasicDataset()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    same_inchi  = molecules[0].to_inchikey(fixed_hydrogens=False)
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        attributes["inchi_key"] = same_inchi
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)

    assert dataset.n_molecules == len(molecules)


def test_basicdataset_add_molecules_single_conformer():
    """
    Test creating a basic dataset.
    """

    dataset = BasicDataset()
    # get some molecules
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # store the molecules in the dataset under a common index
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)

    # now we need to make sure the dataset has been filled.
    assert len(molecules) == dataset.n_molecules
    assert len(molecules) == dataset.n_records

    # now we should remake each molecule and make sure it matches the input
    for mols in zip(dataset.molecules, molecules):
        assert mols[0].is_isomorphic_with(mols[1])


def test_basicdataset_add_molecules_conformers():
    """
    Test adding a molecule with conformers which should each be expanded into their own qcportal.models.Molecule.
    """

    dataset = BasicDataset()
    # create a molecule with multipule conformers
    molecules = Molecule.from_file(get_data('butane_conformers.pdb'))
    # collapse the conformers down
    butane = molecules.pop(0)
    for conformer in molecules:
        butane.add_conformer(conformer.conformers[0])

    assert butane.n_conformers == 7
    # now add to the dataset
    index = butane.to_smiles()
    attributes = get_cmiles(butane)
    dataset.add_molecule(index=index, attributes=attributes, molecule=butane)

    assert dataset.n_molecules == 1
    assert dataset.n_records == 7

    for mol in dataset.molecules:
        assert butane.is_isomorphic_with(mol)
        for i in range(butane.n_conformers):
            assert mol.conformers[i].flatten().tolist() == pytest.approx(butane.conformers[i].flatten().tolist())


def test_basicdataset_coverage_reporter():
    """
    Test generating coverage reports for openforcefield force fields.
    """

    dataset = BasicDataset()
    # get some molecules
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)

    ff = "openff_unconstrained-1.0.0.offxml"
    coverage = dataset.coverage_report(ff)

    assert ff in coverage
    # make sure that every tag has been used
    tags = ["Angles", "Bonds", "ImproperTorsions", "ProperTorsions", "vdW"]
    for tag in tags:
        assert tag in coverage[ff]


def test_basicdataset_add_molecule_no_conformer():
    """
    Test adding molecules with no conformers which should cause the validtor to generate one.
    """

    dataset = BasicDataset()
    ethane = Molecule.from_smiles('CC')
    # add the molecule to the dataset with no conformer
    index = ethane.to_smiles()
    attributes = get_cmiles(ethane)
    dataset.add_molecule(index=index, attributes=attributes, molecule=ethane)

    assert len(dataset.dataset) == 1
    for molecule in dataset.molecules:
        assert molecule.n_conformers != 0


def test_basicdataset_add_molecule_missing_attributes():
    """
    Test adding a molecule to the dataset with a missing cmiles attribute this should raise an error.
    """

    dataset = BasicDataset()
    ethane = Molecule.from_smiles('CC')
    # generate a conformer to make sure this is not rasing an error
    ethane.generate_conformers()
    assert ethane.n_conformers != 0
    index = ethane.to_smiles()
    attributes = {"test": "test"}
    with pytest.raises(DatasetInputError):
        dataset.add_molecule(index=index, attributes=attributes, molecule=ethane)


@pytest.mark.parametrize("file_data", [
    pytest.param(("molecules.smi", "SMI", "to_smiles", {"isomeric": True, "explicit_hydrogens": False}, False), id="smiles"),
    pytest.param(("molecules.inchi", "INCHI", "to_inchi", {}, False), id="inchi"),
    pytest.param(("molecules.inchikey", "inchikey", "to_inchikey", {}, False), id="inchikey"),
    pytest.param(("molecules.hash", "hash", "to_hash", {}, True), id="hash error")
])
def test_basicdataset_molecules_to_file(file_data):
    """
    Test exporting only the molecules in a dataset to file for each of the supported types.
    """

    dataset = BasicDataset()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)
    with temp_directory():
        # if we expect an error
        if not file_data[4]:
            dataset.molecules_to_file(file_name=file_data[0], file_type=file_data[1])

            # now we need to read in the data
            with open(file_data[0]) as molecule_data:
                data = molecule_data.readlines()
                for i, molecule in enumerate(dataset.molecules):
                    # get the function and call it, also pass any extra arguments
                    result = getattr(molecule, file_data[2])(**file_data[3])
                    # now compare the data in the file to what we have calculated
                    assert data[i].strip() == result
        else:
            with pytest.raises(UnsupportedFiletypeError):
                dataset.molecules_to_file(file_name=file_data[0], file_type=file_data[1])


@pytest.mark.parametrize("toolkit_data", [
    pytest.param(("openeye", None, False), id="openeye no highlights"),
    pytest.param(("openeye", None, True), id="openeye with highlights"),
    pytest.param(("rdkit", None, False), id="rdkit, no highlights"),
    pytest.param(("rdkit", None, True), id="rdkit with highlights"),
    pytest.param((None, None, False), id="Openeye by default."),
    pytest.param(("bad_toolkit", ValueError, False), id="Bad toolkit name.")
])
def test_dataset_to_pdf_no_torsions(toolkit_data):
    """
    Test exporting molecules to pdf with no torsions.
    """

    dataset = BasicDataset()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    toolkit, error, highlight = toolkit_data
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        if highlight:
            first_dihedral = list(molecule.propers)[0]
            # get the atomic numbers
            dihedrals = [tuple([atom.molecule_atom_index for atom in first_dihedral])]
        else:
            dihedrals = None
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, dihedrals=dihedrals)

    with temp_directory():
        # try and export the pdf file
        if error:
            with pytest.raises(error):
                dataset.visualize(file_name="molecules.pdf", toolkit=toolkit)
        else:
            dataset.visualize(file_name="molecules.pdf", toolkit=toolkit)
            

@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"), pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_dataset_export_full_dataset_json(dataset_type):
    """
    Test round tripping a full dataset via json.
    """

    dataset = dataset_type()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        try:
            dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)
        except TypeError:
            dihedrals = [get_dihedral(molecule), ]
            dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, dihedrals=dihedrals)
    with temp_directory():
        dataset.export_dataset("dataset.json")

        dataset2 = dataset_type.parse_file("dataset.json")

        assert dataset.n_molecules == dataset2.n_molecules
        assert dataset.n_records == dataset2.n_records
        assert dataset.dataset == dataset.dataset
        assert dataset.metadata == dataset2.metadata


@pytest.mark.parametrize("dataset_type", [
    pytest.param((BasicDataset, OptimizationDataset), id="BasicDataset to OptimizationDataset"),
    pytest.param((OptimizationDataset, BasicDataset), id="OptimizationDataset to BasicDataSet"),
    pytest.param((BasicDataset, TorsiondriveDataset), id="BasicDataSet to TorsiondriveDataset"),
    pytest.param((OptimizationDataset, TorsiondriveDataset), id="OptimizationDataset to TorsiondriveDataset"),
])
def test_dataset_export_full_dataset_json_mixing(dataset_type):
    """
    Test round tripping a full dataset via json from one type to another this should fail as the dataset_types do not
    match.
    """

    dataset = dataset_type[0]()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        dihedrals = [get_dihedral(molecule), ]
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, dihedrals=dihedrals)
    with temp_directory():
        dataset.export_dataset("dataset.json")

        with pytest.raises(ValidationError):
            dataset2 = dataset_type[1].parse_file("dataset.json")


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"), pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_dataset_export_dict(dataset_type):
    """
    Test making a new dataset from the dict of another of the same type.
    """

    dataset = dataset_type()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        dihedrals = [get_dihedral(molecule), ]
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, dihedrals=dihedrals)

    # add one failure
    fail = Molecule.from_smiles("C")
    dataset.filter_molecules(molecules=[fail, ],
                             component_name="TestFailure",
                             component_description={"name": "TestFailure"},
                             component_provenance={"test": "v1.0"})

    dataset2 = dataset_type(**dataset.dict())

    assert dataset.n_molecules == dataset2.n_molecules
    assert dataset.n_records == dataset2.n_records
    assert dataset.metadata == dataset2.metadata
    for record in dataset.dataset.keys():
        assert record in dataset2.dataset


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"),
    pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_dataset_export_json(dataset_type):
    """
    Test that the json serialisation works.
    """

    dataset = dataset_type()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        dihedrals = [get_dihedral(molecule), ]
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, dihedrals=dihedrals)

    # add one failure
    fail = Molecule.from_smiles("C")
    dataset.filter_molecules(molecules=[fail, ],
                             component_name="TestFailure",
                             component_description={"name": "TestFailure"},
                             component_provenance={"test": "v1.0"})


    # try parse the json string to build the dataset
    dataset2 = dataset_type.parse_raw(dataset.json())
    assert dataset.n_molecules == dataset2.n_molecules
    assert dataset.n_records == dataset2.n_records
    for record in dataset.dataset.keys():
        assert record in dataset2.dataset


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"),
    pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
@pytest.mark.parametrize("compression", [
    pytest.param("xz", id="lzma"),
    pytest.param("bz2", id="bz2"),
    pytest.param("gz", id="gz"),
    pytest.param("", id="No compression")
])
def test_dataset_roundtrip_compression(dataset_type, compression):
    """
    Test that the json serialisation works.
    """

    dataset = dataset_type()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = get_cmiles(molecule)
        dihedrals = [get_dhiedral(molecule), ]
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, dihedrals=dihedrals)

    # add one failure
    fail = Molecule.from_smiles("C")
    dataset.filter_molecules(molecules=[fail, ],
                             component_name="TestFailure",
                             component_description={"name": "TestFailure"},
                             component_provenance={"test": "v1.0"})

    with temp_directory():
        # export the dataset with compression
        dataset2 = dataset_type.parse_raw(dataset.json())
        assert dataset.n_molecules == dataset2.n_molecules
        assert dataset.n_records == dataset2.n_records
        for record in dataset.dataset.keys():
            assert record in dataset2.dataset


@pytest.mark.parametrize("basis_data", [
    pytest.param(({"method": "ani1x", "basis": None, "program": "torchani"}, {"P"}, True), id="Ani1x with Error"),
    pytest.param(({"method": "ani1ccx", "basis": None, "program": "torchani"}, {"C", "H", "N"}, False), id="Ani1ccx Pass"),
    pytest.param(({"method": "b3lyp-d3bj", "basis": "dzvp", "program": "psi4"}, {"C", "H", "O"}, False), id="DZVP psi4 convert Pass"),
    pytest.param(({"method": "hf", "basis": "6-311++G", "program": "psi4"}, {"Br", "C", "O", "N"}, True), id="6-311++G Error"),
    pytest.param(({"method": "hf", "basis": "def2-qzvp", "program": "psi4"}, {"H", "C", "B", "N", "O", "F", "Cl", "Si", "P", "S", "I", "Br"}, False), id="Def2-QZVP Pass"),
    pytest.param(({"method": "wb97x-d", "basis": "aug-cc-pV(5+d)Z", "program": "psi4"}, {"I", "C", "H"}, True), id="aug-cc-pV(5+d)Z Error"),
    pytest.param(({"method": "openff-1.0.0", "basis": "smirnoff", "program": "openmm"}, {"C", "H", "N"}, False), id="Smirnoff Pass"),
    pytest.param(({"method": "GFN2-xTB", "basis": None, "program": "xtb"}, {"C", "N", "Se"}, False), id="XTB Pass"),
    pytest.param(({"method": "uff", "basis": None, "program": "rdkit"}, {"C", "N", "O"}, False), id="Rdkit UFF Pass")
])
def test_basis_coverage_single(basis_data):
    """
    Make sure that the datasets can work out if the elements in the basis are covered.
    """

    qc_spec, elements, error = basis_data
    dataset = BasicDataset(metadata={"elements": elements})
    dataset.add_qc_spec(**qc_spec, spec_name="default", spec_description="testing the basis", overwrite=True)

    if error:
        with pytest.raises(MissingBasisCoverageError):
            dataset._get_missing_basis_coverage(raise_errors=error)
    else:

        assert bool(dataset._get_missing_basis_coverage(raise_errors=error)["default"]) is False


def test_basis_coverage_multiple():
    """
    Test getting the basis coverage for multiple specs.
    """

    # make a dataset covered by xtb, psi4, rdkit, smirnoff, but not ani
    dataset = BasicDataset(metadata={"elements": {"I", "C", "H", "N", "O"}})
    specs = [
        {"method": "hf", "basis": "DZVP", "program": "psi4"},
        {"method": "gfn2xtb", "basis": None, "program": "xtb"},
        {"method": "openff-1.0.0", "basis": "smirnoff", "program": "openmm"},
        {"method": "uff", "basis": None, "program": "rdkit"},
        {"method": "ani1ccx", "basis": None, "program": "torchani"}
    ]
    # clear out all qcspecs
    dataset.clear_qcspecs()
    for spec in specs:
        dataset.add_qc_spec(**spec, spec_name=spec["program"], spec_description="testing basis report")

    # get the coverage
    basis_coverage = dataset._get_missing_basis_coverage(raise_errors=False)
    # all should pass but ani
    ani_report  = basis_coverage.pop("torchani")
    assert ani_report == {"I"}
    # make sure the rest are all empty
    for report in basis_coverage.values():
        assert report == set()


def test_remove_qcspec():
    """
    Test removing qcspecs.
    """
    dataset = BasicDataset()
    dataset.remove_qcspec("default")
    assert dataset.qc_specifications == {}


@pytest.mark.parametrize("qc_spec", [
    pytest.param(({"method": "hf", "basis": "DZVP", "program": "bad_program"}, True), id="Bad method Error"),
    pytest.param(({"method": "gfn1xtb", "basis": "bad", "program": "xtb"}, True), id="Bad basis xtb Error"),
    pytest.param(({"method": "ani1x", "basis": "ani1x", "program": "torchani"}, True), id="Bad basis torchani Error"),
    pytest.param(({"method": "parsley", "basis": "smirnoff", "program": "openmm"}, True), id="Bad method smirnoff Error"),
])
def test_adding_qc_specs(qc_spec):
    """
    Test adding different qc_specs to a dataset/factory they are controlled by the mixin so just test that.
    """
    dataset = BasicDataset()
    qc_spec_data, error = qc_spec
    if error:
        with pytest.raises(QCSpecificationError):
            dataset.add_qc_spec(**qc_spec_data, spec_name="test", spec_description="test spec")
    else:
        dataset.add_qc_spec(**qc_spec_data, spec_name="test", spec_description="test spec")
        assert dataset.qc_specifications["test"].dict(include={"method", "basis", "program"}) == qc_spec_data


def test_qcspec_dict():
    """
    Test making a dict with the qcspec.
    """
    dataset = BasicDataset()
    dataset.add_qc_spec(method="b3lyp",
                        basis="dzvp",
                        program="psi4",
                        spec_name="test",
                        spec_description="test qcspec",
                        store_wavefunction="orbitals_and_eigenvalues")
    data = dataset.qc_specifications["test"].dict()
    assert "store_wavefunction" in data
    data = dataset.qc_specifications["test"].dict(exclude={"store_wavefunction"})
    assert "store_wavefunction" not in data


def test_qc_spec_overwrite():
    """
    Test adding new qcspecs and overwiting.
    """
    dataset = BasicDataset()
    # remove all qcspecs
    dataset.clear_qcspecs()
    # raise an error for no specs
    with pytest.raises(QCSpecificationError):
        dataset._check_qc_specs()
    assert dataset.n_qc_specs == 0
    # add a default spec
    dataset.add_qc_spec(method="B3LYP-D3BJ", basis='DZVP',
                        program="psi4",
                        spec_name="default",
                        spec_description="testing overwrite",
                        store_wavefunction="none")
    assert dataset.n_qc_specs == 1
    # now try and add another default spec
    with pytest.raises(QCSpecificationError):
        dataset.add_qc_spec(method="ani1x",
                            basis=None,
                            program="torchani",
                            spec_name="default",
                            spec_description="testing ani1",
                            store_wavefunction="none")
    # now add with overwite
    dataset.add_qc_spec(method="ani1x",
                        basis=None,
                        program="torchani",
                        spec_name="default",
                        spec_description="testing ani1",
                        store_wavefunction="none",
                        overwrite=True)
    # now make sure it has been updated
    spec = dataset.qc_specifications["default"]
    assert spec.method == "ani1x"
    assert spec.basis is None
    assert spec.program == "torchani"


def test_basicdataset_schema():
    """
    Test that producing the schema still works.
    """

    dataset = BasicDataset()
    # make a schema
    schema = dataset.schema()
    assert schema["title"] == dataset.dataset_name
    assert schema["properties"]["compute_tag"]["type"] == "string"


@pytest.mark.parametrize("input_data", [
    pytest.param(("CCC", 0), id="basic core and tag=0"), pytest.param(("CC@@/_-1CC", 10), id="complex core and tag=10")
])
def test_basicdataset_clean_index(input_data):
    """
    Test that index cleaning is working, this checks if an index already has a numeric counter and strips it, this
    allows us to submit molecule indexs that start from a counter other than 0.
    """

    dataset = BasicDataset()

    index = input_data[0] + "-" + str(input_data[1])

    core, counter = dataset._clean_index(index=index)

    assert core == input_data[0]
    assert counter == input_data[1]


def test_basicdataset_clean_index_normal():
    """
    Test that index cleaning works when no numeric counter is on the index this should give back the core and 0 as the
    tag.
    """
    dataset = BasicDataset()
    index = "CCCC"
    core, counter = dataset._clean_index(index=index)
    assert core == index
    assert counter == 0


def test_basicdataset_filtering():
    """
    Test adding filtered molecules to the dataset.
    """

    dataset = BasicDataset()
    molecules = duplicated_molecules(include_conformers=False, duplicates=1)
    # create a filtered result
    component_description = {"component_name": "TestFilter",
                             "component_description": "Test component for filtering molecules"}
    component_provenance = {"test_provenance": "version_1"}
    dataset.filter_molecules(molecules=molecules,
                             component_name="TestFilter",
                             component_description=component_description,
                             component_provenance=component_provenance)

    assert len(molecules) == dataset.n_filtered
    assert dataset.n_components == 1
    # grab the info on the components
    components = dataset.components
    assert len(components) == 1
    component = components[0]
    assert "TestFilter" == component["component_name"]
    assert "version_1" == component["component_provenance"]["test_provenance"]

    # now loop through the molecules to make sure they match
    for mols in zip(dataset.filtered, molecules):
        assert mols[0].is_isomorphic_with(mols[1]) is True


def test_optimizationdataset_qc_spec():
    """
    Test generating the qc spec for optimization datasets.
    """

    dataset = OptimizationDataset(driver="energy")
    qc_spec = dataset.get_qc_spec("default", keyword_id="0")
    assert qc_spec.keywords == "0"
    tags = ['program', "method", "basis", "driver"]
    for tag in tags:
        assert tag in qc_spec.dict()
    # make sure the driver was set back to gradient
    assert qc_spec.driver == "gradient"


def test_torsiondrivedataset_torsion_indices():
    """
    Test adding molecules to a torsiondrive dataset with incorrect torsion indices.
    """

    dataset = TorsiondriveDataset()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)

    for molecule in molecules:
        with pytest.raises(DihedralConnectionError):
            index = molecule.to_smiles()
            attributes = get_cmiles(molecule)
            dataset.add_molecule(index=index, molecule=molecule, attributes=attributes, dihedrals=[(0, 1, 1, 1)])
