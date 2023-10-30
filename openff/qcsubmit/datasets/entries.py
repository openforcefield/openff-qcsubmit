"""
All of the individual dataset entry types are defined here.
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import openff.toolkit.topology as off
import qcelemental as qcel
import qcelemental.models
from openff.units import unit
from pydantic import Field, validator
from typing_extensions import Literal

from openff.qcsubmit.common_structures import (
    DatasetConfig,
    MoleculeAttributes,
    TDSettings,
)
from openff.qcsubmit.constraints import Constraints, ConstraintType
from openff.qcsubmit.exceptions import ConstraintError, DihedralConnectionError
from openff.qcsubmit.utils.smirnoff import (
    combine_openff_molecules,
    split_openff_molecule,
)
from openff.qcsubmit.validators import (
    check_connectivity,
    check_constraints,
    check_improper_connection,
    check_linear_torsions,
    check_torsion_connection,
)


class DatasetEntry(DatasetConfig):
    """
    A basic data class to construct the datasets which holds any information about the molecule and options used in
    the qcarchive calculation.

    Note:
        * ``extras`` are passed into the qcelemental.models.Molecule on creation.
        * any extras that should passed to the calculation like extra constrains should be passed to ``keywords``.
    """

    index: str = Field(
        ...,
        description="The index name the molecule will be stored under in QCArchive. Note that if multipule geometries are provided the index will be augmented with a value indecating the conformer number so -0, -1.",
    )
    initial_molecules: List[qcel.models.Molecule] = Field(
        ...,
        description="A list of QCElemental Molecule objects which contain the geometries to be used as inputs for the calculation.",
    )
    attributes: MoleculeAttributes = Field(
        ...,
        description="The complete set of required cmiles attributes for the molecule.",
    )
    extras: Optional[Dict[str, Any]] = Field(
        {},
        description="Any extra information that should be injected into the QCElemental models before being submited like the cmiles information.",
    )
    keywords: Optional[Dict[str, Any]] = Field(
        {},
        description="Any extra keywords that should be used in the QCArchive calculation should be passed here.",
    )

    def __init__(self, off_molecule: Optional[off.Molecule] = None, **kwargs):
        """
        Init the dataclass handling conversions of the molecule first.
        This is needed to make sure the extras are passed into the qcschema molecule.
        """

        extras = kwargs.get("extras", {})
        # a place holder to check for multiple components
        molecule_ids, charges = None, None
        # if we get an off_molecule we need to convert it
        if off_molecule is not None:
            split_molecules = split_openff_molecule(molecule=off_molecule)
            if len(split_molecules) > 1:
                # recombine the molecule in the correct ordering
                off_molecule = combine_openff_molecules(split_molecules)
                # we need to remake the attributes for the new ordering
                kwargs["attributes"] = MoleculeAttributes.from_openff_molecule(
                    molecule=off_molecule
                )

            molecule_ids = [
                list(sorted(ids))
                for ids in nx.connected_components(off_molecule.to_networkx())
            ]
            charges = [
                sum(
                    [
                        off_molecule.atoms[atom].formal_charge.m_as(
                            unit.elementary_charge
                        )
                        for atom in graph
                    ]
                )
                for graph in molecule_ids
            ]
            if off_molecule.n_conformers == 0:
                off_molecule.generate_conformers(n_conformers=1)
            schema_mols = [
                off_molecule.to_qcschema(conformer=conformer, extras=extras)
                for conformer in range(off_molecule.n_conformers)
            ]
            kwargs["initial_molecules"] = schema_mols

        super().__init__(**kwargs)

        # now we need to process all of the initial molecules to make sure the cmiles is present
        # and force c1 symmetry, we also need to split the molecule into fragments if there are multiple present
        initial_molecules = []
        for mol in self.initial_molecules:
            extras = mol.extras or {}
            extras[
                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
            ] = self.attributes.canonical_isomeric_explicit_hydrogen_mapped_smiles
            identifiers = mol.identifiers or {}
            mol_data = mol.dict()
            mol_data["extras"] = extras
            mol_data["identifiers"] = identifiers
            # put into strict c1 symmetry
            mol_data["fix_symmetry"] = "c1"
            # add fragment information if we have multiple components
            if molecule_ids is not None and len(molecule_ids) > 1:
                mol_data["fragments"] = molecule_ids
                mol_data["fragment_charges"] = charges
            initial_molecules.append(qcel.models.Molecule.parse_obj(mol_data))
        # try and parse the schema to make sure the fragment order is correct
        qcel.molparse.from_schema(molschema=initial_molecules[0].dict(), verbose=True)
        # now assign the new molecules
        self.initial_molecules = initial_molecules

    def get_off_molecule(self, include_conformers: bool = True) -> off.Molecule:
        """Build and openforcefield.topology.Molecule representation of the input molecule.

        Parameters:
            include_conformers: If `True` all of the input conformers are included else they are dropped.
        """

        molecule = self.attributes.to_openff_molecule()
        molecule.name = self.index
        if include_conformers:
            for conformer in self.initial_molecules:
                geometry = unit.Quantity(np.array(conformer.geometry), unit.bohr)
                molecule.add_conformer(geometry.to(unit.angstrom))
        return molecule


class OptimizationEntry(DatasetEntry):
    """
    An optimization dataset specific entry class which can handle constraints.
    """

    constraints: Constraints = Field(
        Constraints(),
        description="Any constraints which should be used during an optimization.",
    )

    def __init__(self, off_molecule: Optional[off.Molecule] = None, **kwargs):
        """
        Here we handle the constraints before calling the super.
        """
        # if the constraints are in the keywords move them out for validation
        if "constraints" in kwargs["keywords"]:
            constraint_dict = kwargs["keywords"].pop("constraints")
            constraints = Constraints(**constraint_dict)
            kwargs["constraints"] = constraints

        super().__init__(off_molecule, **kwargs)
        # validate any constraints being added
        check_constraints(
            constraints=self.constraints,
            molecule=self.get_off_molecule(include_conformers=False),
        )

    def add_constraint(
        self,
        constraint: Literal["set", "freeze"],
        constraint_type: ConstraintType,
        indices: List[int],
        bonded: bool = True,
        **kwargs,
    ) -> None:
        """
        Add new constraint of the given type.

        Parameters:
            constraint: The major type of constraint, freeze or set
            constraint_type: the constraint sub type, angle, distance etc
            indices: The atom indices the constraint should be placed on
            bonded: If the constraint is intended to be put a bonded set of atoms
            kwargs: Any extra information needed by the constraint, for the set class they need a value `value=float`
        """
        if constraint.lower() == "freeze":
            self.constraints.add_freeze_constraint(
                constraint_type=constraint_type, indices=indices, bonded=bonded
            )
        elif constraint.lower() == "set":
            self.constraints.add_set_constraint(
                constraint_type=constraint_type,
                indices=indices,
                bonded=bonded,
                **kwargs,
            )
        else:
            raise ConstraintError(
                f"The constraint {constraint} is not available please chose from freeze or set."
            )
        # run the constraint check
        check_constraints(
            constraints=self.constraints,
            molecule=self.get_off_molecule(include_conformers=False),
        )

    @property
    def formatted_keywords(self) -> Dict[str, Any]:
        """
        Format the keywords with the constraints values.
        """
        import copy

        if self.constraints.has_constraints:
            constraints = self.constraints.dict()
            keywords = copy.deepcopy(self.keywords)
            keywords["constraints"] = constraints
            return keywords
        else:
            return self.keywords


class TorsionDriveEntry(OptimizationEntry):
    """
    A Torsiondrive dataset specific class which can check dihedral indices and store torsiondrive specific settings with built in validation.
    """

    dihedrals: List[Tuple[int, int, int, int]] = Field(
        ...,
        description="The list of dihedrals that should be driven, currently only 1D or 2D torsions are supported.",
    )
    keywords: Optional[TDSettings] = Field(
        TDSettings(),
        description="The torsiondrive keyword settings which can be used to overwrite the general global settings used in the dataset allowing for finner control.",
    )

    # we do not yet support multi component torsion drives so validate
    # we have to define the validation this way due to pydantic
    # <https://pydantic-docs.helpmanual.io/usage/validators/#subclass-validators-and-each_item>
    @validator("initial_molecules")
    def _check_conectivity(
        cls, molecules: List[qcelemental.models.Molecule]
    ) -> List[qcelemental.models.Molecule]:
        for mol in molecules:
            check_connectivity(mol)
        return molecules

    @property
    def formatted_keywords(self) -> Dict[str, Any]:
        """Format the keywords with constraints."""

        if self.constraints.has_constraints:
            keywords = self.keywords.additional_keywords
            keywords["constraints"] = self.constraints.dict()
            return keywords
        else:
            return self.keywords.additional_keywords

    def __init__(self, off_molecule: Optional[off.Molecule] = None, **kwargs):
        super().__init__(off_molecule, **kwargs)
        # now validate the torsions check proper first
        off_molecule = self.get_off_molecule(include_conformers=False)

        # now validate the dihedrals
        for torsion in self.dihedrals:
            # check for linear torsions
            check_linear_torsions(torsion, off_molecule)
            try:
                check_torsion_connection(torsion=torsion, molecule=off_molecule)
            except DihedralConnectionError:
                # if this fails as well raise
                try:
                    check_improper_connection(improper=torsion, molecule=off_molecule)
                except DihedralConnectionError:
                    raise DihedralConnectionError(
                        f"The dihedral {torsion} for molecule {off_molecule} is not a valid"
                        f" proper/improper torsion."
                    )


class FilterEntry(DatasetConfig):
    """
    A basic data class that contains information on components run in a workflow and the associated molecules which were
    removed by it.
    """

    component: str = Field(
        ...,
        description="The name of the component ran, this should be one of the components registered with qcsubmit.",
    )
    component_settings: Dict[str, Any] = Field(
        ...,
        description="The run time settings of the component used to filter the molecules.",
    )
    component_provenance: Dict[str, str] = Field(
        ...,
        description="A dictionary of the version information of all dependencies of the component.",
    )
    molecules: List[str]

    def __init__(self, off_molecules: List[off.Molecule] = None, **kwargs):
        """
        Init the dataclass handling conversions of the molecule first.
        """
        if off_molecules is not None:
            molecules = [
                molecule.to_smiles(isomeric=True, explicit_hydrogens=True)
                for molecule in off_molecules
            ]
            kwargs["molecules"] = molecules

        super().__init__(**kwargs)

    def add_molecule(self, molecule: off.Molecule) -> None:
        """
        Add a molecule to this filter.
        """
        self.molecules.append(
            molecule.to_smiles(isomeric=True, explicit_hydrogens=True)
        )
