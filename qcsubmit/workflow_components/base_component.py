from typing import List, Dict
import abc
from pydantic import BaseModel, validator
from openforcefield.topology import Molecule
from openforcefield.utils.toolkits import RDKitToolkitWrapper, OpenEyeToolkitWrapper

from qcsubmit.datasets import ComponentResult


class CustomWorkflowComponent(BaseModel, abc.ABC):
    """
    This is an abstract base class which should be used to create all workflow components, following the design of this
    class should allow users to easily create new work flow components with out needing to change much of the dataset
    factory code
    """

    component_name: str
    component_description: str
    component_fail_message: str

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    @staticmethod
    @abc.abstractmethod
    def is_available() -> bool:
        """
        This method should identify if the component can be used by checking if the requirements are available.

        Returns:
            `True` if the component can be used else `False`
        """
        ...

    @abc.abstractmethod
    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        This is the main feature of the workflow component which should accept a molecule, perform the component action
        and then return the

        Parameters:
            molecules: The list of molecules to be processed by this component.

        Returns:
            An instance of the [ComponentResult][qcsubmit.datasets.ComponentResult]
            class which handles collecting together molecules that pass and fail
            the component
        """
        ...

    @abc.abstractmethod
    def provenance(self) -> Dict:
        """
        This function should detail the programs with version information and procedures called during activation
        of the workflow component.

        Returns:
            A dictionary containing the information about the component and the functions called.
        """
        ...

    def _create_result(self) -> ComponentResult:
        """
        A helpful method to build to create the component result with the required information.

        Returns
        -------
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instantiated with the required information.
        """

        result = ComponentResult(component_name=self.component_name, component_description=self.dict(),
                                 component_provenance=self.provenance())

        return result

    def fail_molecule(self, molecule: Molecule, component_result: ComponentResult) -> None:
        """
        A method to fail a molecule.

        Parameters:
            molecule: The instance of the molecule to be failed.
            component_result: The [ComponentResult][qcsubmit.datasets.ComponentResult] instance that the molecule should
                be added to.
        """

        component_result.filter_molecule(molecule)


class ToolkitValidator(BaseModel):
    """
    A pydantic mixin class that adds toolkit settings and validation along with provenance information.

    Note:
        The provenance information and toolkit settings are handled by the
        [ToolkitValidator][qcsubmit.workflow_components.base_component.ToolkitValidator] mixin.
    """

    toolkit: str = "openeye"
    _toolkits: Dict = {"rdkit": RDKitToolkitWrapper, "openeye": OpenEyeToolkitWrapper}

    @validator("toolkit")
    def _check_toolkit(cls, toolkit):
        """
        Make sure that toolkit is one of the supported types in the OFFTK.
        """
        if toolkit not in cls._toolkits.keys():
            raise ValueError(
                f"The requested toolkit ({toolkit}) is not support by the OFFTK. "
                f"Please chose from {cls._toolkits.keys()}."
            )
        else:
            return toolkit

    def provenance(self) -> Dict:
        """
        This component calls the OFFTK to perform the task and logs information on the backend toolkit used.

        Returns:
            A dictionary containing the version information about the backend toolkit called to perform the task.
        """

        import openforcefield

        provenance = {"OpenforcefieldToolkit": openforcefield.__version__}
        if self.toolkit == "rdkit":
            import rdkit

            provenance["rdkit"] = rdkit.__version__

        elif self.toolkit == "openeye":
            import openeye

            provenance["openeye"] = openeye.__version__

        return provenance

    @staticmethod
    def is_available():
        """
        Check if any of the requested backend toolkits can be used.
        """

        for toolkit in ToolkitValidator._toolkits.values():
            if toolkit.is_available():
                return True
        else:
            return False
