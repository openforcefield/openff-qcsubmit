"""
Components that aid with Fragmentation of molecules.
"""
from typing import Dict, List, Optional, Union

from openforcefield.topology import Molecule
from pydantic import validator

from ..common_structures import TorsionIndexer
from ..datasets import ComponentResult
from ..serializers import deserialize
from .base_component import CustomWorkflowComponent, ToolkitValidator


class WBOFragmenter(ToolkitValidator, CustomWorkflowComponent):
    """
    Fragment molecules using the WBO fragmenter class of the fragmenter module.

    Atrributes:
        threshold. float, default=0.03
            The WBO threshold to be used when comparing

    """

    component_name = "WBOFragmenter"
    component_description = (
        "Fragment a molecule across all rotatble bonds using the WBO fragmenter."
    )
    component_fail_message = "The molecule could not be fragmented correctly."

    threshold: float = 0.03
    keep_non_rotor_ring_substituents: bool = False
    functional_groups: Optional[Union[bool, str]] = None
    heuristic: str = "path_length"
    include_parent: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._skip_unique_check: bool = False  # This component creates new molecules
        self._processes: Union[
            int, None
        ] = None  # This component uses an expensive calculation
        self._cache: bool = False

    @validator("heuristic")
    def check_heuristic(cls, heuristic):
        """
        Make sure the heuristic is valid.
        """

        allowed_heuristic = ["path_length", "wbo"]
        if heuristic.lower() not in allowed_heuristic:
            raise ValueError(
                f"The requested heuristic must be either path_length or wbo."
            )
        else:
            return heuristic.lower()

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

        try:
            import fragmenter
            import openeye

            return True

        except ImportError:
            return False

    def _apply_init(self, result: ComponentResult) -> None:

        self._cache = self.is_available()

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
        available = self._cache

        result = self._create_result()

        if not available:
            for molecule in molecules:
                result.filter_molecule(molecule)
            return result

        from fragmenter import fragment

        for molecule in molecules:
            # not having a conformer can cause issues
            if len(molecule.conformers) == 0:
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
                    heuristic=self.heuristic,
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
                self.fail_molecule(molecule=molecule, component_result=result)

        return result

    def provenance(self) -> Dict:
        """
        Collect the toolkit information and add the fragmenter version information.
        """

        import fragmenter

        provenance = super().provenance()

        provenance["fragmenter"] = fragmenter.__version__

        return provenance
