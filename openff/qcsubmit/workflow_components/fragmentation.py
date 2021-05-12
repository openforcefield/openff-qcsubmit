"""
Components that aid with Fragmentation of molecules.
"""
from typing import Dict, List, Optional, Union

from openff.toolkit.topology import Molecule
from pydantic import Field, validator
from qcelemental.util import which_import
from typing_extensions import Literal

from openff.qcsubmit.common_structures import ComponentProperties, TorsionIndexer
from openff.qcsubmit.datasets import ComponentResult
from openff.qcsubmit.serializers import deserialize
from openff.qcsubmit.workflow_components.base_component import (
    CustomWorkflowComponent,
    ToolkitValidator,
)


class WBOFragmenter(ToolkitValidator, CustomWorkflowComponent):
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
        description="If any non rotor ring substituents should be kept during the fragmentation resulting in smaller fragments.",
    )
    functional_groups: Optional[Union[bool, str]] = Field(
        None,
        description="The path to the yaml/json file containing a list of functional group types to be considered during fragmentation. Supplying None will cause fragmenter to use"
        "its own predefined list.",
    )
    include_parent: bool = Field(
        False,
        description="If the parent molecule should also be included in the output.",
    )

    @classmethod
    def description(cls) -> str:
        return (
            "Fragment a molecule across all rotatable bonds using the WBO fragmenter."
        )

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule could not be fragmented correctly."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=True)

    @validator("functional_groups")
    def check_functional_groups(cls, functional_group):
        """
        Check the functional groups which can be passed as a file name or as a dictionary are valid.

        Note:
            This check could be quite fragile.
        """
        if functional_group is None or functional_group is False:
            return functional_group

        elif isinstance(functional_group, str):
            fgroups = deserialize(functional_group)
            # simple check on the smarts
            for smarts in fgroups.values():
                if "[" not in smarts:
                    raise ValueError(
                        f"Some functional group smarts were not valid {smarts}."
                    )

            return functional_group

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if fragmenter can be imported.
        """
        openeye = which_import(
            ".oechem",
            raise_error=True,
            return_bool=True,
            package="openeye",
            raise_msg="Please install via `conda install openeye-toolkits -c openeye`.",
        )
        fragmenter = which_import(
            "fragmenter",
            raise_error=True,
            return_bool=True,
            raise_msg="Please install via `conda install fragmenter -c omnia`.",
        )

        return openeye and fragmenter

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Fragment the molecules using the WBOFragmenter.

        Parameters:
            molecules: The list of molecules which should be processed by this component.

        Note:
            * If the input molecule fails fragmentation it will be fail this component and be removed even when
            `include_parent` is set to true.
            * When a molecule can not be fragmented to meet the wbo threshold the parent is likely to be included in the
            dataset.
            *
        """
        from fragmenter import fragment

        result = self._create_result()

        for molecule in molecules:
            # not having a conformer can cause issues
            if molecule.n_conformers == 0:
                molecule.generate_conformers(n_conformers=1)

            if self.include_parent:
                result.add_molecule(molecule)

            fragment_factory = fragment.WBOFragmenter(
                molecule=molecule.to_openeye(),
                functional_groups=self.functional_groups,
                verbose=False,
            )

            try:
                fragment_factory.fragment(
                    threshold=self.threshold,
                    keep_non_rotor_ring_substituents=self.keep_non_rotor_ring_substituents,
                )

                # we need to store the central bond which was fragmented around
                # to make sure this is the bond we torsiondrive around
                fragmets_dict = fragment_factory.to_torsiondrive_json()

                # check we have fragments
                if fragmets_dict:

                    for fragment_data in fragmets_dict.values():
                        frag_mol = Molecule.from_mapped_smiles(
                            mapped_smiles=fragment_data["identifiers"][
                                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                            ]
                        )
                        torsion_index = tuple(fragment_data["dihedral"][0])
                        # this is stored back into the molecule and will be used when generating the cmiles tags latter
                        torsion_tag = TorsionIndexer()
                        torsion_tag.add_torsion(torsion=torsion_index)
                        frag_mol.properties["dihedrals"] = torsion_tag
                        result.add_molecule(frag_mol)

                # if we have no fragments and we dont want the parent then we failed to fragment
                elif not fragmets_dict and not self.include_parent:
                    result.filter_molecule(molecule)

            except (RuntimeError, ValueError):
                # this will catch cmiles errors for molecules with undefined stero
                result.filter_molecule(molecule)

        return result

    def provenance(self) -> Dict:
        """
        Collect the toolkit information and add the fragmenter version information.
        """

        import fragmenter

        provenance = super().provenance()

        provenance["fragmenter"] = fragmenter.__version__

        return provenance
