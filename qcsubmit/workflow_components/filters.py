"""
File containing the filters workflow components.
"""
from typing import Dict, List, Union

import openforcefield
from openforcefield.topology import Molecule
from simtk import unit

from .base_component import CustomWorkflowComponet
from qcsubmit.datasets import ComponentResult


class MolecularWeightFilter(CustomWorkflowComponet):
    """
    Filters molecules based on the minimum and maximum allowed molecular weights.
    """

    componet_name = "MolecularWeightFilter"
    componet_descripton = "Molecules are filtered based on the allowed molecular weights."
    componet_fail_message = "Molecule weight was not in the specified region."

    minimum_weight: int = 130  # values taken from the base settings of the openeye blockbuster filter
    maximum_weight: int = 781

    def apply(self, molecules: List[Molecule]) -> ComponentResult:

        result = ComponentResult()

        for molecule in molecules:
            total_weight = sum([atom.element.mass.value_in_unit(unit.daltons) for atom in molecule.atoms])

            if self.minimum_weight < total_weight < self.maximum_weight:
                result.add_molecule(molecule)
            else:
                self.fail_molecule(molecule=molecule, component_result=result)

        return result

    def provenance(self) -> Dict:

        provenance = {'toolkit': openforcefield.__version__}

        return provenance


class ElementFilter(CustomWorkflowComponet):
    """
    Filter the molecules based on the allowed elements.
    """

    componet_name = "ElementFilter"
    componet_descripton = "Filter out molecules who contain elements not in the allowed element list"
    componet_fail_message = "Molecule contained elements not in the allowed elements list"

    allowed_elements: List[Union[int, str]] = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']

    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Filter out molecules who contain elements not included in the allowed_elements list.
        """
        from simtk.openmm.app import Element

        result = ComponentResult()
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
        Generate the provenance for the operation, here we only use OpenMM simtk unit.
        """

        from simtk import openmm
        provenance = {'simtk_openmm': openmm.__version__}

        return provenance


