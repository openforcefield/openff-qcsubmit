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

    dataset_type = registered_datasets.get(raw_data["dataset_type"].lower(), None)
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
        dataset_type = dataset.__fields__["dataset_type"].default.lower()

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


register_dataset(BasicDataset)
register_dataset(OptimizationDataset)
register_dataset(TorsiondriveDataset)
