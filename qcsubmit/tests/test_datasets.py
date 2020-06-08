"""
Unit test for the vairous dataset classes in the package.
"""

import numpy as np
import pytest
from pydantic import ValidationError
from simtk import unit

from openforcefield.topology import Molecule
from qcsubmit.datasets import (
    BasicDataset,
    ComponentResult,
    OptimizationDataset,
    TorsiondriveDataset,
)
from qcsubmit.exceptions import DatasetInputError, MissingBasisCoverageError
from qcsubmit.testing import temp_directory
from qcsubmit.utils import get_data
from qcsubmit. common_structures import TorsionIndexer


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


def test_componentresult_deduplication_torsions_same_bond_same_coords():
    """
    Make sure that the same rotatable bond is not highlighted more than once when deduplicating molecules.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})

    molecules = [Molecule.from_file(get_data("methanol.sdf"), 'sdf')] * 3
    methanol_dihedrals = [(5, 1, 0, 2), (5, 1, 0, 3), (5, 1, 0, 4)]
    for molecule, dihedral in zip(molecules, methanol_dihedrals):
        torsion_indexer = TorsionIndexer()
        torsion_indexer.add_torsion(torsion=dihedral, scan_range=None)
        molecule.properties["dihedrals"] = torsion_indexer
        result.add_molecule(molecule)

    # now make sure only one dihedral is selected
    assert len(result.molecules) == 1
    assert result.molecules[0].properties["dihedrals"].n_torsions == 1
    # this checks the bond has been ordered
    assert (0, 1) in result.molecules[0].properties["dihedrals"].torsions
    print(result.molecules[0].properties["dihedrals"])


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


def test_componentresult_deduplication_torsions_1d():
    """
    Make sure that any torsion index results are correctly transferred when deduplicating molecules.
    """

    result = ComponentResult(component_name="Test deduplication", component_description={},
                             component_provenance={})

    duplicates = 2
    molecules = duplicated_molecules(include_conformers=False, duplicates=duplicates)

    for molecule in molecules:
        torsion_indexer = TorsionIndexer()
        torsion_indexer.add_torsion(torsion=tuple(np.random.randint(low=0, high=7, size=4).tolist()),
                                    scan_range=tuple(np.random.randint(low=-165, high=180, size=2).tolist()))
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
    molecules = duplicated_molecules(include_conformers=False, duplicates=duplicates)

    for molecule in molecules:
        torsion_indexer = TorsionIndexer()
        torsion_indexer.add_torsion(torsion=tuple(np.random.randint(low=0, high=7, size=4).tolist()),
                                    scan_range=tuple(np.random.randint(low=-165, high=180, size=2)))

        torsion_indexer.add_double_torsion(torsion1=tuple(np.random.randint(low=0, high=7, size=4).tolist()),
                                           torsion2=tuple(np.random.randint(low=0, high=7, size=4).tolist()),
                                           scan_range1=tuple(np.random.randint(low=-165, high=180, size=2)),
                                           scan_range2=tuple(np.random.randint(low=-165, high=180, size=2)))

        molecule.properties["dihedrals"] = torsion_indexer

        result.add_molecule(molecule)

    for molecule in result.molecules:
        assert "dihedrals" in molecule.properties
        assert molecule.properties["dihedrals"].n_torsions == duplicates
        assert molecule.properties["dihedrals"].n_double_torsions == duplicates

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
    pytest.param(BasicDataset, id="BasicDataset"), pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_Dataset_exporting_same_type(dataset_type):
    """
    Test making the given dataset from the json of another instance of the same dataset type.
    """

    with temp_directory():
        dataset = dataset_type(method="test method")
        dataset.export_dataset('dataset.json')

        dataset2 = dataset_type.parse_file('dataset.json')
        assert dataset2.method == "test method"
        assert dataset.metadata == dataset2.metadata


def test_BasicDataset_add_molecules_single_conformer():
    """
    Test creating a basic dataset.
    """

    dataset = BasicDataset()
    # get some molecules
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # store the molecules in the dataset under a common index
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = {"canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(mapped=True),
                      "test_tag": "test"}
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)

    # now we need to make sure the dataset has been filled.
    assert len(molecules) == dataset.n_molecules
    assert len(molecules) == dataset.n_records

    # now we should remake each molecule and make sure it matches the input
    for mols in zip(dataset.molecules, molecules):
        assert mols[0].is_isomorphic_with(mols[1])


def test_BasicDataset_add_molecules_conformers():
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
    attributes = {"canonical_isomeric_explicit_hydrogen_mapped_smiles": butane.to_smiles(mapped=True),
                  "test_tag": "test"}
    dataset.add_molecule(index=index, attributes=attributes, molecule=butane)

    assert dataset.n_molecules == 1
    assert dataset.n_records == 7

    for mol in dataset.molecules:
        assert butane.is_isomorphic_with(mol)
        for i in range(butane.n_conformers):
            assert mol.conformers[i].flatten().tolist() == pytest.approx(butane.conformers[i].flatten().tolist())


def test_BasicDataset_coverage_reporter():
    """
    Test generating coverage reports for openforcefield force fields.
    """

    dataset = BasicDataset()
    # get some molecules
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = {"canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(mapped=True),
                      "test_tag": "test"}
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)

    ff = "openff_unconstrained-1.0.0.offxml"
    coverage = dataset.coverage_report([ff])

    assert ff in coverage
    # make sure that every tag has been used
    tags = ["Angles", "Bonds", "ImproperTorsions", "ProperTorsions", "vdW"]
    for tag in tags:
        assert tag in coverage[ff]


def test_Basicdataset_add_molecule_no_conformer():
    """
    Test adding molecules with no conformers which should cause the validtor to generate one.
    """

    dataset = BasicDataset()
    ethane = Molecule.from_smiles('CC')
    # add the molecule to the dataset with no conformer
    index = ethane.to_smiles()
    attributes = {"canonical_isomeric_explicit_hydrogen_mapped_smiles": ethane.to_smiles(mapped=True),
                  "test_tag": "test"}
    dataset.add_molecule(index=index, attributes=attributes, molecule=ethane)

    assert len(dataset.dataset) == 1
    for molecule in dataset.molecules:
        assert molecule.n_conformers != 0


def test_Basicdataset_add_molecule_missing_attributes():
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
    pytest.param(("molecules.smi", "SMI", "to_smiles"), id="smiles"), pytest.param(("molecules.inchi", "INCHI", "to_inchi"), id="inchi"),
    pytest.param(("molecules.inchikey", "inchikey", "to_inchikey"), id="inchikey")
])
def test_Basicdataset_molecules_to_file(file_data):
    """
    Test exporting only the molecules in a dataset to file for each of the supported types.
    """

    dataset = BasicDataset()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = {"canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(mapped=True),
                      "canonical_isomeric_smiles": molecule.to_smiles(isomeric=True),
                      "standard_inchi": molecule.to_inchi(),
                      "inchi_key": molecule.to_inchikey()}
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)
    with temp_directory():
        dataset.molecules_to_file(file_name=file_data[0], file_type=file_data[1])

        # now we need to read in the data
        with open(file_data[0]) as molecule_data:
            data = molecule_data.readlines()
            for i, molecule in enumerate(dataset.molecules):
                # get the function and call it
                result = getattr(molecule, file_data[2])()
                # now compare the data in the file to what we have calculated
                assert data[i].strip() == result


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"), pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_Dataset_export_full_dataset_json(dataset_type):
    """
    Test round tripping a full dataset via json.
    """

    dataset = dataset_type()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = {"canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(mapped=True),
                      "canonical_smiles": molecule.to_smiles(),
                      "standard_inchi": molecule.to_inchi(),
                      "inchi_key": molecule.to_inchikey()}
        try:
            dataset.add_molecule(index=index, attributes=attributes, molecule=molecule)
        except TypeError:
            dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, atom_indices=(0, 1, 2, 3))
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
def test_Dataset_export_full_dataset_json_mixing(dataset_type):
    """
    Test round tripping a full dataset via json from one type to another this should fail as the dataset_types do not
    match.
    """

    dataset = dataset_type[0]()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = {"canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(mapped=True),
                      "canonical_smiles": molecule.to_smiles(),
                      "standard_inchi": molecule.to_inchi(),
                      "inchi_key": molecule.to_inchikey()}

        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, dihedrals=[(0, 1, 2, 3)])
    with temp_directory():
        dataset.export_dataset("dataset.json")

        with pytest.raises(ValidationError):
            dataset2 = dataset_type[1].parse_file("dataset.json")


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"), pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_Dataset_export_dict(dataset_type):
    """
    Test making a new dataset from the dict of another of the same type.
    """

    dataset = dataset_type()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = {"canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(mapped=True),
                      "canonical_smiles": molecule.to_smiles(),
                      "standard_inchi": molecule.to_inchi(),
                      "inchi_key": molecule.to_inchikey()}

        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, dihedrals=[(0, 1, 2, 3)])

    dataset2 = dataset_type(**dataset.dict())

    assert dataset.n_molecules == dataset2.n_molecules
    assert dataset.n_records == dataset2.n_records
    assert dataset.metadata == dataset2.metadata
    assert dataset.json() == dataset2.json()


@pytest.mark.parametrize("dataset_type", [
    pytest.param(BasicDataset, id="BasicDataset"), pytest.param(OptimizationDataset, id="OptimizationDataset"),
    pytest.param(TorsiondriveDataset, id="TorsiondriveDataset")
])
def test_Basicdataset_export_json(dataset_type):
    """
    Test that the json serialisation works.
    """

    dataset = dataset_type()
    molecules = duplicated_molecules(include_conformers=True, duplicates=1)
    # add them to the dataset
    for molecule in molecules:
        index = molecule.to_smiles()
        attributes = {"canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(mapped=True),
                      "canonical_smiles": molecule.to_smiles(),
                      "standard_inchi": molecule.to_inchi(),
                      "inchi_key": molecule.to_inchikey()}
        dataset.add_molecule(index=index, attributes=attributes, molecule=molecule, dihedrals=[(0, 1, 2, 3)])


    # try parse the json string to build the dataset
    dataset2 = dataset_type.parse_raw(dataset.json())
    assert dataset.n_molecules == dataset2.n_molecules
    assert dataset.n_records == dataset2.n_records
    assert dataset.json() == dataset2.json()


@pytest.mark.parametrize("basis_data", [
    pytest.param(("ani1x", None, {"P"}, "torchani", True), id="Ani1x with Error"),
    pytest.param(("ani1ccx", None, {"C", "H", "N"}, "torchani", False), id="Ani1ccx Pass"),
    pytest.param(("b3lyp-d3bj", "dzvp", {"C", "H", "O"}, "psi4", False), id="DZVP psi4 convert Pass"),
    pytest.param(("hf", "6-311++G", {"Br", "C", "O", "N"}, "psi4", True), id="6-311++G Error"),
    pytest.param(("hf", "def2-qzvp", {"H", "C", "B", "N", "O", "F", "Cl", "Si", "P", "S", "I", "Br"}, "psi4", False), id="Def2-QZVP Pass"),
    pytest.param(("wb97x-d", "aug-cc-pV(5+d)Z", {"I", "C", "H"}, "psi4", True), id="aug-cc-pV(5+d)Z Error")
])
def test_basis_coverage(basis_data):
    """
    Make sure that the datasets can work out if the elements in the basis are covered.
    """

    method, basis, elements, program, error = basis_data
    dataset = BasicDataset(method=method, basis=basis, metadata={"elements": elements}, program=program)

    if error:
        with pytest.raises(MissingBasisCoverageError):
            dataset._get_missing_basis_coverage(raise_errors=error)
    else:

        assert bool(dataset._get_missing_basis_coverage(raise_errors=error)) is False


def test_Basicdataset_schema():
    """
    Test that producing the schema still works.
    """

    dataset = BasicDataset()
    # make a schema
    schema = dataset.schema()
    assert schema["title"] == dataset.dataset_name
    assert schema["properties"]["method"]["type"] == "string"


@pytest.mark.parametrize("input_data", [
    pytest.param(("CCC", 0), id="basic core and tag=0"), pytest.param(("CC@@/_-1CC", 10), id="complex core and tag=10")
])
def test_Basicdataset_clean_index(input_data):
    """
    Test that index cleaning is working, this checks if an index already has a numeric counter and strips it, this
    allows us to submit molecule indexs that start from a counter other than 0.
    """

    dataset = BasicDataset()

    index = input_data[0] + "-" + str(input_data[1])

    core, counter = dataset._clean_index(index=index)

    assert core == input_data[0]
    assert counter == input_data[1]


def test_Basicdataset_clean_index_normal():
    """
    Test that index cleaning works when no numeric counter is on the index this should give back the core and 0 as the
    tag.
    """
    dataset = BasicDataset()
    index = "CCCC"
    core, counter = dataset._clean_index(index=index)
    assert core == index
    assert counter == 0


def test_Basicdataset_filtering():
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


def test_Optimizationdataset_qc_spec():
    """
    Test generating the qc spec for optimization datasets.
    """

    dataset = OptimizationDataset(program="test_program", method="test_method", basis="test_basis",
                                  driver="energy")
    qc_spec = dataset.get_qc_spec(keyword_id="0")
    assert qc_spec.keywords == "0"
    tags = ['program', "method", "basis", "driver"]
    for tag in tags:
        assert tag in qc_spec.dict()
    # make sure the driver was set back to gradient
    assert qc_spec.driver == "gradient"
