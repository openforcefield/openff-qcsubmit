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

# The QCS Dataset.type field was originally a mapping from QCS datasets to QCF collection type.
# As of the QCF "next" (0.50) release, QCF has its own dataset classes. However,
# for reverse-compatibility (being able to load serialized files from old versions of QCS), the QCS
# dataset classes will continue using the original types internally, and will just convert to the
# new QCF types using the following dict when needed.
legacy_qcsubmit_ds_type_to_next_qcf_ds_type = {
    "DataSet": "singlepoint",
    "OptimizationDataset": "optimization",
    "TorsionDriveDataset": "torsiondrive",
}


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

    from openff.qcsubmit.datasets import legacy_qcsubmit_ds_type_to_next_qcf_ds_type

    # make sure all specs are gone
    dataset.clear_qcspecs()
    qcf_ds_type = legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type]
    ds = client.get_dataset(qcf_ds_type, dataset.dataset_name)
    metadata = ds.metadata
    if "elements" in metadata:
        dataset.metadata = metadata

    if qcf_ds_type == "singlepoint":
        if not dataset.metadata.elements:
            # now grab the elements
            elements = set()
            # molecules = ds.get_molecules()
            # for index in molecules.index:
            # TODO: Does iterate_entries guarantee an output order?
            for index, entry in enumerate(ds.iterate_entries()):
                mol = entry.molecule
                # mol = molecules.loc[index].molecule
                elements.update(mol.symbols)
            dataset.metadata.elements = elements
        # now we need to add each ran spec
        # for history in ds.data.history:
        for spec_name, specification in ds.specifications.items():
            program = specification.specification.program
            method = specification.specification.method
            basis = specification.specification.basis
            spec_name = specification.name
            if program.lower() != "dftd3":
                # the composition planner breaks the validation
                dataset.add_qc_spec(
                    method=method,
                    basis=basis,
                    program=program,
                    spec_name=spec_name,
                    spec_description="basic dataset spec",
                    overwrite=True,
                )
    elif qcf_ds_type == "optimization":
        # we have the opt or torsiondrive
        if not dataset.metadata.elements:
            elements = set()
            for entry in ds.iterate_entries():
                formula = entry.attributes["molecular_formula"]
                # use regex to parse the formula
                match = re.findall("[A-Z][a-z]?|\d+|.", formula)
                for element in match:
                    if not element.isnumeric():
                        elements.add(element)
            dataset.metadata.elements = elements
        # now add the specs
        for spec_name, spec in ds.specifications.items():
            dataset.add_qc_spec(
                method=spec.specification.qc_specification.method,
                basis=spec.specification.qc_specification.basis,
                program=spec.specification.qc_specification.program,
                spec_name=spec_name,
                spec_description=spec.description,
                store_wavefunction=spec.specification.qc_specification.protocols.wavefunction.value,
            )
    elif qcf_ds_type == "torsiondrive":
        # we have the opt or torsiondrive
        if not dataset.metadata.elements:
            elements = set()
            for entry in ds.iterate_entries():
                formula = entry.attributes["molecular_formula"]
                # use regex to parse the formula
                match = re.findall("[A-Z][a-z]?|\d+|.", formula)
                for element in match:
                    if not element.isnumeric():
                        elements.add(element)
            dataset.metadata.elements = elements
        # now add the specs
        for spec_name, spec in ds.specifications.items():
            dataset.add_qc_spec(
                method=spec.specification.optimization_specification.qc_specification.method,
                basis=spec.specification.optimization_specification.qc_specification.basis,
                program=spec.specification.optimization_specification.qc_specification.program,
                spec_name=spec_name,
                spec_description=spec.description,
                store_wavefunction=spec.specification.optimization_specification.qc_specification.protocols.wavefunction.value,
            )
    else:
        assert 0
    return dataset


register_dataset(BasicDataset)
register_dataset(OptimizationDataset)
register_dataset(TorsiondriveDataset)
