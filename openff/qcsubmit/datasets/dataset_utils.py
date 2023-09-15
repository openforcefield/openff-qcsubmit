"""
A set of utility functions to help with loading datasets.
"""
from typing import Any, Dict, List, Union

from openff.qcsubmit.datasets.datasets import (
    BasicDataset,
    OptimizationDataset,
    TorsiondriveDataset,
)
from openff.qcsubmit.exceptions import DatasetRegisterError, InvalidDatasetError
from openff.qcsubmit.serializers import deserialize

registered_datasets: Dict[str, Any] = {}


def load_dataset(data: Union[str, Dict]) -> "BasicDataset":
    """
    Create a new instance dataset from the file or dict of the dataset. This removes the need of knowing what the dataset type is.

    Parameters:
        data: The file path or dict of the dataset from which we should load the data.

    Raises:
        DatasetRegisterError: If no registered dataset can load the given dataset type.

    Returns:
        An instance of the correct dataset loaded with the data.
    """
    if isinstance(data, str):
        # load the file
        raw_data = deserialize(data)
    else:
        raw_data = data

    dataset_type = registered_datasets.get(raw_data["type"].lower(), None)
    if dataset_type is not None:
        return dataset_type(**raw_data)
    else:
        raise DatasetRegisterError(
            f"No registered dataset can load the type {dataset_type}."
        )


def register_dataset(dataset: Any, replace: bool = False) -> None:
    """
    Register a dataset with qcsubmit making it easy to auto load the model from file.

    Parameters:
        dataset: The dataset class that should be registered.
        replace: If the new dataset should replace any other dataset of the same type

    Raises:
        InvalidDatasetError: If the dataset is not a valid sub class of the basic dataset model
        DatasetRegisterError: If a dataset of this type has already been registered
    """

    if issubclass(dataset, BasicDataset):
        dataset_type = dataset.__fields__["type"].default.lower()

        if dataset_type not in registered_datasets or (
            dataset_type in registered_datasets and replace
        ):
            registered_datasets[dataset_type] = dataset
        else:
            raise DatasetRegisterError(
                f"A dataset was already registered with the type {dataset_type}, to replace this use the `replace=True` flag."
            )

    else:
        raise InvalidDatasetError(
            f"Dataset {dataset} rejected as it is not a valid sub class of the BasicDataset."
        )


def list_datasets() -> List[str]:
    """
    Returns:
        A list of all of the currently registered dataset classes.
    """
    return list(registered_datasets.values())


def update_specification_and_metadata(
    dataset: Union[BasicDataset, OptimizationDataset, TorsiondriveDataset], client
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
    ds = client.get_collection(dataset.type, dataset.dataset_name)
    metadata = ds.data.metadata
    if "elements" in metadata:
        dataset.metadata = metadata

    if dataset.type == "DataSet":
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
            if program.lower() != "dftd3":
                # the composition planner breaks the validation
                dataset.add_qc_spec(
                    method=method,
                    basis=basis,
                    program=program,
                    spec_name=spec,
                    spec_description="basic dataset spec",
                    overwrite=True,
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
                optimization_trajectory=spec.qc_spec.protocols.trajectory.value,
            )

    return dataset


register_dataset(BasicDataset)
register_dataset(OptimizationDataset)
register_dataset(TorsiondriveDataset)
