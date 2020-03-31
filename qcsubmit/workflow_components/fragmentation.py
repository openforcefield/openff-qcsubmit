"""
Components that aid with Fragmentation of molecules.
"""

from typing import List, Union, Dict
from fragmenter import fragment
from .base_component import CustomWorkflowComponent, ToolkitValidator
from ..datasets import ComponentResult
from pydantic import validator
from openforcefield.topology import Molecule


class WBOFragmenter(ToolkitValidator, CustomWorkflowComponent):
    """
    Fragment molecules using the WBO fragmenter class of the fragmenter module.

    Atrributes:
        threshold. float, default=0.03
            The WBO threshold to be used when comparing

    """

    component_name = "WBOFragmenter"
    component_description = "Fragment a molecule across all rotatble bonds using the WBO fragmenter."
    component_fail_message = "The molecule could not fragmented correctly."

    threshold: float = 0.03
    keep_non_rotor_ring_substituents: bool = False
    functional_groups: Union[str, Dict, List] = None
    heuristic: str = 'path_length'

    @validator('heuristic')
    def check_heuristic(cls, heuristic):
        """
        Make sure the heuristic is valid.
        """

        allowed_heuristic = ['path_length', 'wbo']
        if heuristic.lower() not in allowed_heuristic:
            raise ValueError(f'The requested heuristic must be either path_length or wbo.')
        else:
            return heuristic.lower()

    @validator('functional_groups', each_item=True)
    def check_functional_groups(cls, functional_group):
        """
        Check the functional groups which can be passed as a file name or as a dictionary are valid.
        """
        if functional_group is None:
            return functional_group
        elif isinstance(functional_group, str):
            pass

    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Fragment the molecules using the WBOFragmenter.

        Parameters:
            molecules:

        Important:
            The input molecule will be removed from the dataset after fragmentation.
        """
        pass
