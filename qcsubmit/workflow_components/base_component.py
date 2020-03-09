from typing import List, Dict
import abc
from pydantic import BaseModel
from openforcefield.topology import Molecule


class ComponentMissingError(Exception):
    pass


class CustomWorkflowComponet(BaseModel, abc.ABC):
    """
    This is an abstract base class which should be used to create all workflow componets, following the desgin of this
    class should allow users to easily create new work flow components with out needing to change much of the dataset
    factory code
    """

    componet_name: str
    componet_descripton: str
    componet_fail_message: str

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    @abc.abstractmethod
    def apply(self, molecule: Molecule):
        """
        This is the main feature of the workflow component which should accept a molecule, perform the component action
        and then return the

        Prameters
        ----------

        molecule: List[openforcefield.topology.Molecules]:
            The list fof molecules to be processed by this component.

        Returns
        -------
        pass_molecules:

        fail_molecules:
        """
        ...

    @abc.abstractmethod
    def provenance(self) -> Dict:
        """
        This function should detail the programs with version information and procedures called during activation
        of the workflow component.

        Returns
        -------
        provenance: Dict
            A dictionary containing the information about the component. Each program should have its own entry
        """
        ...
