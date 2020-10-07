from typing import Dict, List, Union

from openforcefield import topology as off
from openforcefield.utils.toolkits import (
    RDKitToolkitWrapper,
    UndefinedStereochemistryError,
)

from qcsubmit.datasets import BasicDataset, OptimizationDataset, TorsiondriveDataset


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


def check_missing_stereo(molecule: off.Molecule) -> bool:
    """
    Get if the given molecule has missing stereo by round trip and catching stereo errors.

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


def update_specification_and_metadata(
    dataset: Union["BasicDataset", "OptimizationDataset", "TorsiondriveDataset"], client
) -> Union[BasicDataset, OptimizationDataset, TorsiondriveDataset]:
    """
    For the given dataset update the metadata and specifications using data from an archive instance.

    Parameters:
        dataset: The dataset which should be updated this should have no qc_specs and contain the name of the dataset
        client: The archive client instance
    """
    import re

    # make sure all specs are gone
    dataset.clear_qcspecs()
    ds = client.get_collection(dataset.dataset_type, dataset.dataset_name)
    metadata = ds.data.metadata
    if "elements" in metadata:
        dataset.metadata = metadata

    if dataset.dataset_type == "DataSet":
        if not dataset.metadata.elements:
            # now grab the elements
            elements = set()
            molecules = ds.get_molecules()
            for index in molecules.index:
                mol = molecules.loc[index].molecule
                elements.update(mol.symbols)
            dataset.metadata.elements = elements
        # now we need to add each ran spec
        for history in ds.data.history:
            _, program, method, basis, spec = history
            dataset.add_qc_spec(
                method=method,
                basis=basis,
                program=program,
                spec_name=spec,
                spec_description="basic dataset spec",
            )
    else:
        # we have the opt or torsiondrive
        if not dataset.metadata.elements:
            elements = set()
            for record in ds.data.records.values():
                formula = record.attributes["molecular_formula"]
                # use regex to parse the formula
                match = re.findall("[A-Z][a-z]?|\d+|.", formula)
                for element in match:
                    if not element.isnumeric():
                        elements.add(element)
            dataset.metadata.elements = elements
        # now add the specs
        for spec in ds.data.specs.values():
            dataset.add_qc_spec(
                method=spec.qc_spec.method,
                basis=spec.qc_spec.basis,
                program=spec.qc_spec.program,
                spec_name=spec.name,
                spec_description=spec.description,
                store_wavefunction=spec.qc_spec.protocols.wavefunction.value,
            )

    return dataset
