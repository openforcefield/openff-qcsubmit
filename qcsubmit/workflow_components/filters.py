"""
File containing the filters workflow components.
"""
from typing import Dict, List, Union, Optional
from pydantic import validator

import openforcefield
from openforcefield.topology import Molecule
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils.structure import get_molecule_parameterIDs

from .base_component import CustomWorkflowComponent
from qcsubmit.datasets import ComponentResult


class MolecularWeightFilter(CustomWorkflowComponent):
    """
    Filters molecules based on the minimum and maximum allowed molecular weights.

    Attributes:
        component_name: The name of component.
        component_description: A short description of the component.
        component_fail_message: The message logged when a molecule fails this component.
        minimum_weight: The minimum allowed molecular weight of a molecule.
        maximum_weight: The maximum allowed molecular weight of a molecule.
    """

    component_name = "MolecularWeightFilter"
    component_description = "Molecules are filtered based on the allowed molecular weights."
    component_fail_message = "Molecule weight was not in the specified region."

    minimum_weight: int = 130  # values taken from the base settings of the openeye blockbuster filter
    maximum_weight: int = 781

    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        The common entry point of all workflow components which applies the workflow component to the given list of
        molecules.

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        from simtk import unit

        result = self._create_result()
        for molecule in molecules:
            total_weight = sum([atom.element.mass.value_in_unit(unit.daltons) for atom in molecule.atoms])

            if self.minimum_weight < total_weight < self.maximum_weight:
                result.add_molecule(molecule)
            else:
                self.fail_molecule(molecule=molecule, component_result=result)

        return result

    def provenance(self) -> Dict:
        """
        Generate version information for all of the software used during the running of this component.

        Returns:
            A dictionary of all of the software used in the component along wither their version numbers.

        Important:
            The simtk unit module has no version information so the version of OpenMM is given instead.
        """

        from simtk import openmm

        provenance = {"OpenforcefieldToolkit": openforcefield.__version__, "openmm_units": openmm.__version__}

        return provenance

    @staticmethod
    def is_available() -> bool:
        """
        This filter requires only the basic modules and should always be available.
        """
        return True


class ElementFilter(CustomWorkflowComponent):
    """
    Filter the molecules based on a list of allowed elements.

    Attributes:
        component_name: The name of component.
        component_description: A short desciption of the component.
        component_fail_message: The message logged when a molecule fails this component.
        allowed_elements: A list of atomic symbols or atomic numbers which are allowed passed the filter.

    Note:
        The `allowed_elements` attribute can take a list of either symbols or atomic numbers and will resolve them to a
        common internal format as required.

    Example:
        Using atomic symbols or atomic numbers in components.

        ```python
        >>> from qcsubmit import workflow_components
        >>> efil = workflow_components.ElementFilter()
        # set the allowed elements to H,C,N,O
        >>> efil.allowed_elements = ['H', 'C', 'N', 'O']
        >>> efil.allowed_elements = [1, 6, 7, 8]
        ```
    """

    component_name = "ElementFilter"
    component_description = "Filter out molecules who contain elements not in the allowed element list"
    component_fail_message = "Molecule contained elements not in the allowed elements list"

    allowed_elements: List[Union[int, str]] = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

    @validator("allowed_elements", each_item=True)
    def check_allowed_elements(cls, element: Union[str, int]) -> Union[str, int]:
        """
        Check that each item can be cast to a valid element.

        Parameters:
            element: The element that should be checked.

        Raises:
            ValueError: If the element number or symbol passed could not be converted into a valid element.
        """
        from simtk.openmm.app import Element

        if isinstance(element, int):
            return element
        else:
            try:
                e = Element.getBySymbol(element)
                return element
            except KeyError:
                raise KeyError(f"An element could not be determined from symbol {element}, please eneter symbols only.")

    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        The common entry point of all workflow components which applies the workflow component to the given list of
        molecules.

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """
        from simtk.openmm.app import Element

        result = self._create_result()

        # First lets convert the allowed_elements list to ints as this is what is stored in the atom object
        _allowed_elements = [
            Element.getBySymbol(ele).atomic_number if isinstance(ele, str) else ele for ele in self.allowed_elements
        ]

        # now apply the filter
        for molecule in molecules:
            for atom in molecule.atoms:
                if atom.atomic_number not in _allowed_elements:
                    self.fail_molecule(molecule=molecule, component_result=result)
                    break
            else:
                result.add_molecule(molecule)

        return result

    @staticmethod
    def is_available() -> bool:
        """
        This should always be available as it only requires basic packages.
        """
        return True

    def provenance(self) -> Dict:
        """
        Generate version information for all of the software used during the running of this component.

        Returns:
            A dictionary of all of the software used in the component along wither their version numbers.

        Note:
            The element class in OpenMM is used to match the elements so the OpenMM version is given.
        """

        from simtk import openmm

        provenance = {"openmm_elements": openmm.__version__}

        return provenance


class CoverageFilter(CustomWorkflowComponent):
    """
    Filters molecules based on the requested forcefield coverage.

    Important:
        The ids supplied to the respective group are the ids that are allowed, if `None` is passed all ids are allowed.

    Atributes:
        allowed_ids: The list of parameter ids that we want to actively pass the filter.
        filtered_ids: The list of parameter ids that we want to actively filter out and fail the filter.

    Note:
        If a molecule has any id in the allowed_ids and not in the filtered ids it is passed. Any molecule with a
        parameter in both sets is failed.

    Important:
        A value of None in a list will let all molecules through.
    """

    component_name = "CoverageFilter"
    component_description = "Filter the molecules based on the requested FF allowed parameters."
    component_fail_message = "The molecule was typed with disallowed parameters."

    allowed_ids: Optional[List[str]] = None
    filtered_ids: Optional[List[str]] = None
    forcefield: str = "openff_unconstrained-1.0.0.offxml"

    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Apply the filter to the list of molecules to remove any molecules typed by an id that is not allowed, i.e. not
        included in the allowed list.

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        # pass all of the molecules then filter ones that have elements that are not allowed
        result = self._create_result()

        # the forcefield we are testing against
        forcefield = ForceField(self.forcefield)
        parameters_by_molecule, parameters_by_ID = get_molecule_parameterIDs(molecules, forcefield)

        # loop through the tags
        if self.filtered_ids is not None:
            for filtered_id in self.filtered_ids:
                for molecule in parameters_by_ID.get(filtered_id, []):
                    self.fail_molecule(molecule, result)

        if self.allowed_ids is not None:
            for pid, molecules in parameters_by_ID.items():
                if pid not in self.allowed_ids:
                    for molecule in molecules:
                        self.fail_molecule(molecule, result)

        return result

    @staticmethod
    def is_available() -> bool:
        """
        This should always be available as it only needs basic packages.
        """
        return True

    def provenance(self) -> Dict:
        """
        Generate version information for all of the software used during the running of this component.

        Returns:
            A dictionary of all of the software used in the component along wither their version numbers.
        """
        import openforcefields

        provenance = {
            "oopenforcefields": openforcefields.__version__,
            "OpenforcefieldToolkit": openforcefield.__version__,
        }

        return provenance
