"""
This file contains common starting structures which can be mixed into datasets, results and factories.
"""
import getpass
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import qcportal as ptl
from pydantic import BaseModel, HttpUrl, constr

from qcsubmit.exceptions import DatasetInputError


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


class ClientHandler:
    """
    This mixin class offers the ability to handle activating qcportal Fractal client instances.
    """

    @staticmethod
    def _activate_client(client) -> ptl.FractalClient:
        """
        Make the fractal client and connect to the requested instance.

        Parameters:
            client: The name of the file containing the client information or the client instance.

        Returns:
            A qcportal.FractalClient instance.
        """

        if isinstance(client, ptl.FractalClient):
            return client
        elif client == "public":
            return ptl.FractalClient()
        else:
            return ptl.FractalClient.from_file(client)


class Metadata(DatasetConfig):
    """
    A general metadata class which is required to be filled in before submitting a dataset to the qcarchive.
    """

    submitter: str = getpass.getuser()
    creation_date: date = datetime.today().date()
    collection_type: Optional[str] = None
    dataset_name: Optional[str] = None
    short_description: Optional[constr(min_length=8, regex="[a-zA-Z]")] = None
    long_description_url: Optional[HttpUrl] = None
    long_description: Optional[constr(min_length=8, regex="[a-zA-Z]")] = None
    elements: Set[str] = set()

    def validate_metadata(self, raise_errors: bool = False) -> Optional[List[str]]:
        """
        Before submitting this function should be called to highlight any incomplete fields.
        """

        empty_fields = []
        for field in self.__fields__:
            attr = getattr(self, field)
            if attr is None:
                empty_fields.append(field)

        if empty_fields and raise_errors:
            raise DatasetInputError(
                f"The metadata has the following incomplete fields {empty_fields}"
            )
        else:
            return empty_fields
