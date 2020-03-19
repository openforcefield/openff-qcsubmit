"""
File containing the filters workflow components.
"""
from typing import Dict, List, Union

import openforcefield
from openforcefield.topology import Molecule

from .base_component import CustomWorkflowComponent
from qcsubmit.datasets import ComponentResult


class MolecularWeightFilter(CustomWorkflowComponent):
    """
    Filters molecules based on the minimum and maximum allowed molecular weights.

    Attributes:
        component_name: The name of component.
        component_description: A short desciption of the component.
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

        result = ComponentResult(component_name=self.component_name,
                                 component_description=self.dict())
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

        provenance = {'toolkit': openforcefield.__version__, 'openmm_units': openmm.__version__}

        return provenance


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

    allowed_elements: List[Union[int, str]] = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']

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

        result = ComponentResult(component_name=self.component_name,
                                 component_description=self.dict())

        # First lets convert the allowed_elements list to ints as this is what is stored in the atom object
        _allowed_elements = [Element.getBySymbol(ele).atomic_number if isinstance(ele, str) else ele for ele in self.allowed_elements]

        # now apply the filter
        for molecule in molecules:
            for atom in molecule.atoms:
                if atom.atomic_number not in _allowed_elements:
                    self.fail_molecule(molecule=molecule, component_result=result)
                    break
            else:
                result.add_molecule(molecule)

        return result

    def provenance(self) -> Dict:
        """
        Generate version information for all of the software used during the running of this component.

        Returns:
            A dictionary of all of the software used in the component along wither their version numbers.

        Note:
            The element class in OpenMM is used to match the elements so the OpenMM version is given.
        """

        from simtk import openmm
        provenance = {'openmm_elements': openmm.__version__}

        return provenance


