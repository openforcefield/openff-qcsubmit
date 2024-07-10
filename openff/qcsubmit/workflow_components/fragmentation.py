"""
Components that aid with Fragmentation of molecules.
"""

from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from openff.toolkit import Molecule
from openff.toolkit.utils import ToolkitRegistry
from qcelemental.util import which_import

from openff.qcsubmit._pydantic import Field, validator
from openff.qcsubmit.common_structures import ComponentProperties
from openff.qcsubmit.utils import get_symmetry_classes, get_symmetry_group, get_torsion
from openff.qcsubmit.validators import check_environments
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

    target_torsion_smarts: Optional[List[str]] = Field(
        None,
        description="The list of SMARTS patterns used to identify central target bonds to fragment around. By default this is any single non-termial bond.",
    )

    _check_smarts = validator(
        "target_torsion_smarts",
        each_item=True,
        allow_reuse=True,
    )(check_environments)

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

    def provenance(self, toolkit_registry: ToolkitRegistry) -> Dict:
        """
        Collect the toolkit information and add the fragmenter version information.
        """

        from openff import fragmenter

        provenance = super().provenance(toolkit_registry=toolkit_registry)

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
            symmetry_classes = get_symmetry_classes(fragment_mol)
            symmetry_group = get_symmetry_group(
                atom_group=(bond.atom1_index, bond.atom2_index),
                symmetry_classes=symmetry_classes,
            )
            torsion = get_torsion(bond)
            torsion_tag = TorsionIndexer()
            torsion_tag.add_torsion(torsion=torsion, symmetry_group=symmetry_group)
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
    heuristic: Literal["path_length", "wbo"] = Field(
        "path_length",
        description="The path fragmenter should take when fragment needs to be grown "
        "out. The options are ``['wbo', 'path_length']``.",
    )

    @classmethod
    def description(cls) -> str:
        return (
            "Fragment a molecule across all rotatable bonds using the WBO fragmenter."
        )

    def _apply(
        self, molecules: List[Molecule], toolkit_registry: ToolkitRegistry
    ) -> ComponentResult:
        """
        Fragment the molecules using the WBOFragmenter.

        Args:
            molecules: The list of molecules which should be processed by this component.
            toolkit_registry: The openff.toolkit.utils.ToolkitRegistry which declares the available toolkits.

        Note:
            * If the input molecule fails fragmentation it will fail this component and be removed.
            * When a molecule can not be fragmented to meet the wbo threshold the parent is likely to be included in the
            dataset.
        """
        from openff.fragmenter.fragment import WBOFragmenter

        result = self._create_result(toolkit_registry=toolkit_registry)

        for molecule in molecules:
            fragment_factory = WBOFragmenter(
                threshold=self.threshold,
                keep_non_rotor_ring_substituents=self.keep_non_rotor_ring_substituents,
                heuristic=self.heuristic,
            )

            try:
                fragment_result = fragment_factory.fragment(
                    molecule=molecule,
                    toolkit_registry=toolkit_registry,
                    target_bond_smarts=self.target_torsion_smarts,
                )
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

    def _apply(
        self, molecules: List[Molecule], toolkit_registry: ToolkitRegistry
    ) -> ComponentResult:
        """
        Fragment the molecules using the PfizerFragmenter.

        Args:
            molecules: The list of molecules which should be processed by this component.
            toolkit_registry: The openff.toolkit.utils.ToolkitRegistry which declares the available toolkits.

        Note:
            * If the input molecule fails fragmentation it will be fail this component and be removed.
        """
        from openff.fragmenter.fragment import PfizerFragmenter

        result = self._create_result(toolkit_registry=toolkit_registry)

        for molecule in molecules:
            fragment_factory = PfizerFragmenter()

            try:
                fragment_result = fragment_factory.fragment(
                    molecule=molecule,
                    toolkit_registry=toolkit_registry,
                    target_bond_smarts=self.target_torsion_smarts,
                )
                self._process_fragments(
                    fragments=fragment_result, component_result=result
                )

            except (RuntimeError, ValueError):
                # this will catch cmiles errors for molecules with undefined stero
                result.filter_molecule(molecule)

        return result


class RECAPFragmenter(ToolkitValidator, CustomWorkflowComponent):
    """
    Fragment the molecules using the RECAP algorithm in rdkit from Lewell et al. JCICS 38 511-522 (1998).

    Note:
        This is not used to identify rotatable torsions it is to split the molecule into chemically realistic
        building blocks
    """

    type: Literal["RECAPFragmenter"] = "RECAPFragmenter"

    @classmethod
    def description(cls) -> str:
        return "Fragment a molecule into building blocks using the RECAP method."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule could not be fragmented."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=True)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Rdkit is installed"""
        rdkit = which_import(
            module=".Chem",
            raise_error=True,
            return_bool=True,
            package="rdkit",
            raise_msg="Please install via `mamba install rdkit -c conda-forge`.",
        )
        return rdkit

    def _apply(
        self, molecules: List[Molecule], toolkit_registry: ToolkitRegistry
    ) -> ComponentResult:
        """
        Fragment the molecules using the RECAP method in rdkit.

        Note:
            Method adapted from
            <https://github.com/SimonBoothroyd/gnn-charge-models/blob/ee02a4426eb14c48bfb30c9894af267510efcac0/data-set-curation/generate-fragments.py>
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, Recap

        result = self._create_result(toolkit_registry=toolkit_registry)

        # define some substructure replacements to cap the molecules
        rd_dummy_replacements = [
            # Handle the special case of -S(=O)(=O)[*] -> -S(=O)(-[O-])
            (Chem.MolFromSmiles("S(=O)(=O)*"), Chem.MolFromSmiles("S(=O)([O-])")),
            # Handle the general case
            (Chem.MolFromSmiles("*"), Chem.MolFromSmiles("[H]")),
        ]

        for molecule in molecules:
            rd_parent = Chem.RemoveAllHs(molecule.to_rdkit())

            leaves = Recap.RecapDecompose(rd_parent).GetLeaves()
            for fragment_node in leaves.values():

                rd_fragment = fragment_node.mol

                for rd_dummy, rd_replacement in rd_dummy_replacements:
                    rd_fragment = AllChem.ReplaceSubstructs(
                        rd_fragment, rd_dummy, rd_replacement, True
                    )[0]
                    # Do a SMILES round-trip to avoid wierd issues with radical formation...
                    rd_fragment = Chem.MolFromSmiles(Chem.MolToSmiles(rd_fragment))

                if Descriptors.NumRadicalElectrons(rd_fragment) > 0:
                    # if the fragment is radical skip it
                    continue

                result.add_molecule(molecule=Molecule.from_rdkit(rd_fragment))

        return result
