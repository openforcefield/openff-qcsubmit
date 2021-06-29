"""
Components that aid with Fragmentation of molecules.
"""
from typing import TYPE_CHECKING, Dict, List

from openff.toolkit.topology import Molecule
from pydantic import Field
from qcelemental.util import which_import
from typing_extensions import Literal

from openff.qcsubmit.common_structures import ComponentProperties
from openff.qcsubmit.utils import get_torsion
from openff.qcsubmit.workflow_components.base_component import (
    CustomWorkflowComponent,
    ToolkitValidator,
)
from openff.qcsubmit.workflow_components.utils import ComponentResult, TorsionIndexer

if TYPE_CHECKING:
    from openff.fragmenter.fragment import FragmentationResult


class FragmenterBase(ToolkitValidator, CustomWorkflowComponent):
    """A common base fragmenter class which handles tagging the targeted bond for torsion driving."""

    type: Literal["FragmenterBase"]

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule could not be fragmented correctly."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=True)

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if fragmenter can be imported.
        """
        toolkit = which_import(
            ".toolkit",
            raise_error=True,
            return_bool=True,
            package="openff",
            raise_msg="Please install via `conda install openff-toolkit -c conda-forge`.",
        )
        fragmenter = which_import(
            ".fragmenter",
            raise_error=True,
            return_bool=True,
            package="openff",
            raise_msg="Please install via `pip install git+https://github.com/openforcefield/fragmenter.git@master`.",
        )

        return toolkit and fragmenter

    def provenance(self) -> Dict:
        """
        Collect the toolkit information and add the fragmenter version information.
        """

        from openff import fragmenter

        provenance = super().provenance()

        provenance["openff-fragmenter"] = fragmenter.__version__

        return provenance

    @classmethod
    def _process_fragments(
        cls, fragments: "FragmentationResult", component_result: ComponentResult
    ):
        """Process the resulting fragments and tag the targeted bonds ready for torsiondrives."""
        from openff.fragmenter.utils import get_atom_index

        for bond_map, fragment in fragments.fragments_by_bond.items():
            fragment_mol = fragment.molecule
            # get the index of the atoms in the fragment
            atom1, atom2 = get_atom_index(fragment_mol, bond_map[0]), get_atom_index(
                fragment_mol, bond_map[1]
            )
            bond = fragment_mol.get_bond_between(atom1, atom2)
            torsion = get_torsion(bond)
            torsion_tag = TorsionIndexer()
            torsion_tag.add_torsion(torsion=torsion)
            fragment_mol.properties["dihedrals"] = torsion_tag
            del fragment_mol.properties["atom_map"]
            component_result.add_molecule(fragment_mol)


class WBOFragmenter(FragmenterBase):
    """
    Fragment molecules using the WBO fragmenter class of the fragmenter module.
    For more information see <https://github.com/openforcefield/fragmenter>.
    """

    type: Literal["WBOFragmenter"] = "WBOFragmenter"
    threshold: float = Field(
        0.03,
        description="The WBO error threshold between the parent and the fragment value, the fragmentation will stop when the difference between the fragment and parent is less than this value.",
    )
    keep_non_rotor_ring_substituents: bool = Field(
        False,
        description="If any non rotor ring substituents should be kept during the fragmentation resulting in smaller fragments when `False`.",
    )

    @classmethod
    def description(cls) -> str:
        return (
            "Fragment a molecule across all rotatable bonds using the WBO fragmenter."
        )

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Fragment the molecules using the WBOFragmenter.

        Args:
            molecules: The list of molecules which should be processed by this component.

        Note:
            * If the input molecule fails fragmentation it will fail this component and be removed.
            * When a molecule can not be fragmented to meet the wbo threshold the parent is likely to be included in the
            dataset.
        """
        from openff.fragmenter.fragment import WBOFragmenter

        result = self._create_result()

        for molecule in molecules:

            fragment_factory = WBOFragmenter(
                threshold=self.threshold,
                keep_non_rotor_ring_substituents=self.keep_non_rotor_ring_substituents,
            )

            try:
                fragment_result = fragment_factory.fragment(molecule=molecule)
                self._process_fragments(
                    fragments=fragment_result, component_result=result
                )

            except (RuntimeError, ValueError):
                # this will catch cmiles errors for molecules with undefined stero
                result.filter_molecule(molecule)

        return result


class PfizerFragmenter(FragmenterBase):
    """The openff.fragmenter implementation of the Pfizer fragmenation method as described here
    (doi: 10.1021/acs.jcim.9b00373)
    """

    type: Literal["PfizerFragmenter"] = "PfizerFragmenter"

    @classmethod
    def description(cls) -> str:
        return "Fragment a molecule across all rotatable bonds using the Pfizer fragmentation scheme."

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Fragment the molecules using the PfizerFragmenter.

        Args:
            molecules: The list of molecules which should be processed by this component.

        Note:
            * If the input molecule fails fragmentation it will be fail this component and be removed.
        """
        from openff.fragmenter.fragment import PfizerFragmenter

        result = self._create_result()

        for molecule in molecules:

            fragment_factory = PfizerFragmenter()

            try:
                fragment_result = fragment_factory.fragment(molecule=molecule)
                self._process_fragments(
                    fragments=fragment_result, component_result=result
                )

            except (RuntimeError, ValueError):
                # this will catch cmiles errors for molecules with undefined stero
                result.filter_molecule(molecule)

        return result
