"""
Components that aid with Fragmentation of molecules.
"""
from typing import List, Union, Dict
from .base_component import CustomWorkflowComponent, ToolkitValidator
from ..datasets import ComponentResult
from pydantic import validator
from openforcefield.topology import Molecule
import json
import yaml


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
    functional_groups: Union[bool, str] = None
    heuristic: str = "path_length"
    _file_readers = {"json": json.load, "yaml": yaml.safe_load_all}

    @validator("heuristic")
    def check_heuristic(cls, heuristic):
        """
        Make sure the heuristic is valid.
        """

        allowed_heuristic = ["path_length", "wbo"]
        if heuristic.lower() not in allowed_heuristic:
            raise ValueError(f"The requested heuristic must be either path_length or wbo.")
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
        elif functional_group:
            return None

        elif isinstance(functional_group, str):
            # if its a file we need to check the smarts inside
            file_type = functional_group.split(".")[-1]
            try:
                fgroups = cls._file_readers[file_type](functional_group)
            except KeyError:
                raise ValueError(
                    f"The given file type is not supported {file_type} please use one of the supported "
                    f"file types {cls._file_readers.keys()}"
                )
            except FileNotFoundError:
                raise FileNotFoundError(f"The functional group file {functional_group} could not be found.")

        else:
            raise ValueError(
                f"The given input is not supported please give the path to a file containing the "
                f"functional group smarts."
            )

        # simple check on the smarts
        for smarts in fgroups.values():
            if "[" not in smarts:
                raise ValueError(f"Some functional group smarts were not valid {smarts}.")
        else:
            return functional_group

    @staticmethod
    def is_available() -> bool:
        """
        Check if fragmenter can be imported.
        """

        try:
            import fragmenter
            return True
        except ImportError:
            return False

    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Fragment the molecules using the WBOFragmenter.

        Parameters:
            molecules:

        Important:
            The input molecule will be removed from the dataset after fragmentation.
        """
        from fragmenter import fragment
        result = ComponentResult(component_name=self.component_name, component_description=self.dict())

        for molecule in molecules:
            fragment_factory = fragment.WBOFragmenter(
                molecule=molecule.to_openeye(), functional_groups=self.functional_groups, verbose=False
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

                for fragment_data in fragmets_dict.values():
                    frag_mol = Molecule.from_mapped_smiles(
                        mapped_smiles=fragment_data["identifiers"]["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
                    )
                    torsion_index = tuple(fragment_data["dihedral"][0])
                    # this is stored back into the molecule and will be used when generating the cmiles tags latter
                    try:
                        # none refers to the scan range and is the same as (-165, 180)
                        frag_mol.properties["torsion_index"][torsion_index] = None
                    except KeyError:
                        frag_mol.properties["torsion_index"] = {torsion_index: None}
                    result.add_molecule(frag_mol)

            except RuntimeError:
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
