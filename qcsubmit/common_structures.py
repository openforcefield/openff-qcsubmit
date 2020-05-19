"""
This file contains common starting structures which can be mixed into datasets, results and factories.
"""
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, Tuple, Optional
import re
import numpy as np
from datetime import datetime, date
import getpass


class DatasetConfig(BaseModel):
    """
    The basic configurations for all datasets.
    """

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = True
        validate_assignment: bool = True
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}


class ResultsConfig(BaseModel):
    """
    A basic config class for results structures.
    """

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = False
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}


class IndexCleaner:
    """
    This class offers the ability to clean a molecule index that already has a numeric tag useful for datasets and
    results.
    """

    @staticmethod
    def _clean_index(index: str) -> Tuple[str, int]:
        """
        Take an index and clean it by checking if it already has an enumerator in it return the core index and any
        numeric tags if no tag is found the tag is set to 0.

        Parameters:
            index: The index for the entry which should be checked, if no numeric tag can be found return 0.

        Returns:
            A tuple of the core index and the numeric tag it starts from.

        Note:
            This function allows the dataset to add more conformers to a molecule set so long as the index the molecule
            is stored under is a new index not in the database for example if 3 conformers for ethane exist then the
            new index should start from 'CC-3'.
        """
        # tags take the form '-no'
        match = re.search("-[0-9]+$", index)
        if match is not None:
            core = index[: match.span()[0]]
            # drop the -
            tag = int(match.group()[1:])
        else:
            core = index
            tag = 0

        return core, tag


class Metadata(DatasetConfig):
    """
    A general metadata class which is required to be filled in before submitting a dataset to the qcarchive.
    """

    submitter: str = getpass.getuser()
    creation_date: date = datetime.today().date()
    collection: str
    dataset_name: str
    description: str
    url: Optional[HttpUrl] = None

