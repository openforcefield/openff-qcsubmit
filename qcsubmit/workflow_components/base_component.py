from typing import List, Dict
import abc
from pydantic import BaseModel
from openforcefield.topology import Molecule

from qcsubmit.datasets import ComponentResult


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
    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        This is the main feature of the workflow component which should accept a molecule, perform the component action
        and then return the

        Prameters
        ----------

        molecule: List[openforcefield.topology.Molecules]:
            The list fof molecules to be processed by this component.

        Returns
        -------
        results: ComponentResult,
            An instance of the componentresult class which handles collecting to gether molecules that pass and fail
            the component
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

    def fail_molecule(self, molecule: Molecule, component_result: ComponentResult):
        """A helpful method to fail a molecule will fill in the reason it failed"""

        component_result.filter_molecule(molecule,
                                         component_name=self.componet_name,
                                         component_description=self.componet_descripton,
                                         reason=self.componet_fail_message)
