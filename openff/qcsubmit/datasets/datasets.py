import abc
import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import qcelemental as qcel
import qcportal as ptl
import tqdm
from openff.toolkit import topology as off
from pydantic import Field, constr, validator
from qcportal.models.common_models import (
    DriverEnum,
    OptimizationSpecification,
    QCSpecification,
)
from simtk import unit
from typing_extensions import Literal

from openff.qcsubmit.common_structures import (
    CommonBase,
    Metadata,
    MoleculeAttributes,
    QCSpec,
    TorsionIndexer,
)
from openff.qcsubmit.constraints import Constraints
from openff.qcsubmit.datasets.entries import (
    DatasetEntry,
    FilterEntry,
    OptimizationEntry,
    TorsionDriveEntry,
)
from openff.qcsubmit.exceptions import (
    DatasetCombinationError,
    MissingBasisCoverageError,
    QCSpecificationError,
    UnsupportedFiletypeError,
)
from openff.qcsubmit.procedures import GeometricProcedure
from openff.qcsubmit.serializers import deserialize, serialize
from openff.qcsubmit.utils import chunk_generator
from openff.qcsubmit.utils.visualize import molecules_to_pdf

if TYPE_CHECKING:  # pragma: no cover
    from qcportal import FractalClient
    from qcportal.collections.collection import Collection
    from qcportal.models.common_models import OptimizationSpecification

C = TypeVar("C", bound="Collection")
E = TypeVar("E", bound=DatasetEntry)


class ComponentResult:
    """
    Class to contain molecules after the execution of a workflow component this automatically applies de-duplication to
    the molecules. For example if a molecule is already in the molecules list it will not be added but any conformers
    will be kept and transferred.


    If a molecule in the molecules list is then filtered it will be removed from the molecules list.
    """

    def __init__(
        self,
        component_name: str,
        component_description: Dict[str, str],
        component_provenance: Dict[str, str],
        molecules: Optional[Union[List[off.Molecule], off.Molecule]] = None,
        input_file: Optional[str] = None,
        input_directory: Optional[str] = None,
        skip_unique_check: Optional[bool] = False,
        verbose: bool = True,
    ):
        """Register the list of molecules to process.

        Args:
            component_name:
                The name of the component that produced this result.
            component_description:
                The dictionary representation of the component which details the function and running parameters.
            component_provenance:
                The dictionary of the modules used and there version number when running the component.
            molecules:
                The list of molecules that have been possessed by a component and returned as a result.
            input_file:
                The name of the input file used to produce the result if not from a component.
            input_directory:
                The name of the input directory which contains input molecule files.
            verbose:
                If the timing information and progress bar should be shown while doing deduplication.
            skip_unique_check:
                Set to True if it is sure that all molecules will be unique in this result
        """

        self._molecules: Dict[str, off.Molecule] = {}
        self._filtered: Dict[str, off.Molecule] = {}
        self.component_name: str = component_name
        self.component_description: Dict = component_description
        self.component_provenance: Dict = component_provenance
        self.skip_unique_check: bool = skip_unique_check

        assert (
            molecules is None or input_file is None
        ), "Provide either a list of molecules or an input file name."

        # if we have an input file load it
        if input_file is not None:
            molecules = off.Molecule.from_file(
                file_path=input_file, allow_undefined_stereo=True
            )
            if not isinstance(molecules, list):
                molecules = [
                    molecules,
                ]

        if input_directory is not None:
            molecules = []
            for file in os.listdir(input_directory):
                # each file could have many molecules in it so combine
                mols = off.Molecule.from_file(
                    file_path=os.path.join(input_directory, file),
                    allow_undefined_stereo=True,
                )
                try:
                    molecules.extend(mols)
                except TypeError:
                    molecules.append(mols)

        # now lets process the molecules and add them to the class
        if molecules is not None:
            for molecule in tqdm.tqdm(
                molecules,
                total=len(molecules),
                ncols=80,
                desc="{:30s}".format("Deduplication"),
                disable=not verbose,
            ):
                self.add_molecule(molecule)

    @property
    def molecules(self) -> List[off.Molecule]:
        """
        Get the list of molecules which can be iterated over.
        """
        return list(self._molecules.values())

    @property
    def filtered(self) -> List[off.Molecule]:
        """
        Get the list of molecule that have been filtered to iterate over.
        """
        return list(self._filtered.values())

    @property
    def n_molecules(self) -> int:
        """
        The number of molecules saved in the result.
        """

        return len(self._molecules)

    @property
    def n_conformers(self) -> int:
        """
        The number of conformers stored in the molecules.
        """

        conformers = sum(
            [molecule.n_conformers for molecule in self._molecules.values()]
        )
        return conformers

    @property
    def n_filtered(self) -> int:
        """
        The number of filtered molecules.
        """
        return len(self._filtered)

    def add_molecule(self, molecule: off.Molecule) -> bool:
        """
        Add a molecule to the molecule list after checking that it is not present already. If it is de-duplicate the
        record and condense the conformers and metadata.

        Args:
            molecule:
                The molecule and its conformers which we should try and add to the result.

        Returns:
            `True` if the molecule is already present and `False` if not.
        """
        # always strip the atom map as it is not preserved in a workflow
        if "atom_map" in molecule.properties:
            del molecule.properties["atom_map"]

        # make a unique molecule hash independent of atom order or conformers
        molecule_hash = molecule.to_inchikey(fixed_hydrogens=True)

        if not self.skip_unique_check and molecule_hash in self._molecules:
            # we need to align the molecules and transfer the coords and properties
            # get the mapping, drop some comparisons to match inchikey
            isomorphic, mapping = off.Molecule.are_isomorphic(
                molecule,
                self._molecules[molecule_hash],
                return_atom_map=True,
                formal_charge_matching=False,
                bond_order_matching=False,
            )
            assert isomorphic is True
            # transfer any torsion indexes for similar fragments
            if "dihedrals" in molecule.properties:
                # we need to transfer the properties; get the current molecule dihedrals indexer
                # if one is missing create a new one
                current_indexer = self._molecules[molecule_hash].properties.get(
                    "dihedrals", TorsionIndexer()
                )

                # update it with the new molecule info
                current_indexer.update(
                    torsion_indexer=molecule.properties["dihedrals"],
                    reorder_mapping=mapping,
                )

                # store it back
                self._molecules[molecule_hash].properties["dihedrals"] = current_indexer

            if molecule.n_conformers != 0:

                # transfer the coordinates
                for conformer in molecule.conformers:
                    new_conformer = np.zeros((molecule.n_atoms, 3))
                    for i in range(molecule.n_atoms):
                        new_conformer[i] = conformer[mapping[i]].value_in_unit(
                            unit.angstrom
                        )

                    new_conf = unit.Quantity(value=new_conformer, unit=unit.angstrom)

                    # check if the conformer is already on the molecule
                    for old_conformer in self._molecules[molecule_hash].conformers:
                        if old_conformer.tolist() == new_conf.tolist():
                            break
                    else:
                        self._molecules[molecule_hash].add_conformer(
                            new_conformer * unit.angstrom
                        )
            else:
                # molecule already in list and coords not present so just return
                return True

        else:
            if molecule.n_conformers == 0:
                # make sure this is a list to avoid errors
                molecule._conformers = []
            self._molecules[molecule_hash] = molecule
            return False

    def filter_molecule(self, molecule: off.Molecule):
        """
        Filter out a molecule that has not passed this workflow component. If the molecule is already in the pass list
        remove it and ensure it is only in the filtered list.

        Args:
            molecule:
                The molecule which should be filtered.
        """

        molecule_hash = molecule.to_inchikey(fixed_hydrogens=True)
        try:
            del self._molecules[molecule_hash]

        except KeyError:
            pass

        finally:
            if molecule not in self._filtered:
                self._filtered[molecule_hash] = molecule

    def __repr__(self):
        return f"ComponentResult(name={self.component_name}, molecules={self.n_molecules}, filtered={self.n_filtered})"

    def __str__(self):
        return f"<ComponentResult name='{self.component_name}' molecules='{self.n_molecules}' filtered='{self.n_filtered}'>"


class _BaseDataset(abc.ABC, CommonBase):
    """
    A general base model for QCSubmit datasets which act as wrappers around a corresponding QFractal collection.
    """

    dataset_name: str = Field(
        ...,
        description="The name of the dataset, this will be the name given to the collection in QCArchive.",
    )
    dataset_tagline: constr(min_length=8, regex="[a-zA-Z]") = Field(
        ...,
        description="The tagline should be a short description of the dataset which will be displayed by the QCArchive client when the collections are listed.",
    )
    type: Literal["_BaseDataset"] = Field(
        "_BaseDataset",
        description="The dataset type corresponds to the type of collection that will be made in QCArchive.",
    )
    description: constr(min_length=8, regex="[a-zA-Z]") = Field(
        ...,
        description="A long description of the datasets purpose and details about the molecules within.",
    )
    metadata: Metadata = Field(
        Metadata(), description="The metadata describing the dataset."
    )
    provenance: Dict[str, str] = Field(
        {},
        description="A dictionary of the software and versions used to generate the dataset.",
    )
    dataset: Dict[str, DatasetEntry] = Field(
        {}, description="The actual dataset to be stored in QCArchive."
    )
    filtered_molecules: Dict[str, FilterEntry] = Field(
        {},
        description="The set of workflow components used to generate the dataset with any filtered molecules.",
    )
    _file_writers = {"json": json.dump}

    def __init__(self, **kwargs):
        """
        Make sure the metadata has been assigned correctly if not autofill some information.
        """

        super().__init__(**kwargs)

        # set the collection type here
        self.metadata.collection_type = self.type
        self.metadata.dataset_name = self.dataset_name

        # some fields can be reused here
        if self.metadata.short_description is None:
            self.metadata.short_description = self.dataset_tagline
        if self.metadata.long_description is None:
            self.metadata.long_description = self.description

    @classmethod
    @abc.abstractmethod
    def _entry_class(cls) -> Type[E]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _generate_collection(self, client: "FractalClient") -> C:
        """Generate the corresponding QCFractal Collection for this Dataset.

        Each QCSubmit Dataset class corresponds to and wraps
        a QCFractal Collection class. This method generates an instance
        of that corresponding Collection, with inputs applied from
        Dataset attributes.

        Args:
            client:
                Client to use for connecting to a QCFractal server instance.

        Returns:
            Collection instance corresponding to this Dataset.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_procedure_spec(self) -> "OptimizationSpecification":
        """Get the procedure spec, if applicable, for this Dataset.

        If the dataset has no concept of procedure specs, this method
        should return `None`.

        Returns:
            Specification for the optimization procedure to perform.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_indices(self, dataset: C) -> List[str]:
        """Shim method to get indices from different Collection types.

        The mechanism for getting indices from different QCFractal Collections
        is inconsistent. This method wraps the required calls to the given
        Collection to yield these indices.

        Parameters
        ----------
        collection : Collection
            Collection instance corresponding to this Dataset.

        Returns
        -------
        indices : List[str]
            Indices from the entries in the Collection.

        """
        pass

    @abc.abstractmethod
    def _compute_kwargs(self, spec: QCSpec, indices: List[str]) -> Dict[str, Any]:
        """Returns a dict giving the full arguments to the Collection's
        `compute` method.

        This requires the compute spec defining the compute operations,
        as well as the set of indices to operate on.

        Parameters
        ----------
        spec : QCSpec
            The method, basis, program, and other parameters for compute execution.
        indices : List[str]
            List of entry indices to apply the compute spec to.

        Returns
        -------
        spec_kwargs : Dict[str, Any]
            A dict giving the full arguments to the compute method of this
            Dataset's corresponding Collection.

        """
        pass

    @abc.abstractmethod
    def _add_entries(self, dataset: C, chunk_size: int) -> Tuple[List[str], int]:
        """Add entries to the Dataset's corresponding Collection.

        This method allows for handling of e.g. generating the index/name for
        the corresponding Collection from each item in `self.dataset`. Since
        each item may feature more than one conformer, appropriate handling
        differs between e.g. `OptimizationDataset` and `TorsiondriveDataset`

        Parameters
        ----------
        collection : Collection
            Collection instance corresponding to this Dataset.
        chunk_size : int
            The max number of entries to submit to the QCFractal Server at a time.
            Increase this number to yield better performance.
            The maximum allowed size is set on a per-server basis as its
            `query_limit`.

        Returns
        -------
        indices : List[str]
            A list of the entry indices added.
        new_entries : int
            The number of new entries added.

        """
        pass

    @abc.abstractmethod
    def _add_entry(
        self, molecule: qcel.models.Molecule, dataset: C, name: str, data: E
    ) -> bool:
        """
        Attempt to add a molecule from the local dataset to the QCArchive instance.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _add_dataset_specification(
        self,
        spec: QCSpec,
        dataset: C,
        procedure_spec: Optional["OptimizationSpecification"] = None,
    ) -> bool:
        """Add the given compute spec to this Datasets's corresponding Collection.

        This will check if a spec under this name has already been added and if it should be overwritten.

        If a specification is already stored under this name in the collection we have options:
            - If a spec with the same name but different details has been added and used we must raise an error to change the name of the new spec
            - If the spec has been added and has not been used then overwrite it.

        Parameters
        ----------
        spec : QCSpec
            The QCSpec we are trying to add to the Collection.
        procedure_spec : OptimizationSpecification
            The procedure spec to add, in this case an `OptimizationSpecification`.
        dataset : Collection
            The Collection to add this compute spec to.


        Returns
        -------
        bool
            `True` if the specification is present in the collection
            and is exactly the same as what we are trying to add.

        Raises
        ------
        QCSpecificationError
            If a specification with the same name is already added to the collection but has different settings.
        """
        pass

    def submit(
        self,
        client: Union[str, "FractalClient"],
        processes: Optional[int] = 1,
        ignore_errors: bool = False,
        verbose: bool = False,
        chunk_size: Optional[int] = None,
    ) -> Dict:
        """
        Submit the dataset to a QCFractal server.

        Args:
            client:
                The name of the file containing the client information or an actual client instance.
            processes:
                Number of processes to use for submission; if ``None``, all available processes will be used.
            ignore_errors:
                If the user wants to submit the compute regardless of errors set this to ``True``.
                Mainly to override basis coverage.
            chunk_size:
                Max number of molecules or molecules+specs to include in individual server calls.
                If ``None``, will use ``query_limit`` from ``client``.
            verbose:
                If progress bars and submission statistics should be printed ``True`` or not ``False``.

        Returns:
            A dictionary of the compute response from the client for each specification submitted.

        Raises:
            MissingBasisCoverageError:
                If the chosen basis set does not cover some of the elements in the dataset.

        """
        if chunk_size is None:
            chunk_size = client.query_limit

        # pre submission checks
        # make sure we have some QCSpec to submit
        self._check_qc_specs()
        # basis set coverage check
        self._get_missing_basis_coverage(raise_errors=(not ignore_errors))

        # get client instance
        target_client = self._activate_client(client)

        # see if collection already exists
        # if so, we'll extend it
        # if not, we'll create a new one
        try:
            collection = target_client.get_collection(self.type, self.dataset_name)
        except KeyError:
            self.metadata.validate_metadata(raise_errors=not ignore_errors)
            collection = self._generate_collection(client=target_client)

        # create specifications
        procedure_spec = self._get_procedure_spec()
        for spec in self.qc_specifications.values():
            self._add_dataset_specification(
                spec=spec, dataset=collection, procedure_spec=procedure_spec
            )

        # add the molecules to the database
        indices, new_entries = self._add_entries(collection, chunk_size)
        if verbose:
            print(f"Number of new entries: {new_entries}/{self.n_records}")

        # if we have no indices, such as with a pure compute submission,
        # then get all of the existing ones and use these
        if not indices:
            indices = self._get_indices(collection)

        responses = {}
        # set up process pool for compute submission
        # if processes == 1, perform in-process, no pool
        if processes is None or processes > 1:
            from multiprocessing.pool import Pool

            with Pool(processes=processes) as pool:

                # add compute specs to the collection
                for spec_name, spec in self.qc_specifications.items():
                    spec_tasks = 0
                    work_list = []
                    for mol_chunk in chunk_generator(indices, chunk_size=chunk_size):
                        spec_kwargs = self._compute_kwargs(spec, mol_chunk)
                        work_list.append(
                            pool.apply_async(collection.compute, **spec_kwargs)
                        )

                    for work in tqdm.tqdm(
                        work_list,
                        total=len(work_list),
                        ncols=80,
                        desc=f"Creating tasks for: {spec_name}",
                        disable=not verbose,
                    ):
                        result = work.get()
                        try:
                            spec_tasks += result
                        except TypeError:
                            spec_tasks += len(result.ids)

                    responses[spec_name] = spec_tasks

        else:
            for spec_name, spec in self.qc_specifications.items():
                spec_tasks = 0
                for mol_chunk in tqdm.tqdm(
                    chunk_generator(indices, chunk_size=chunk_size),
                    total=len(indices) / chunk_size,
                    ncols=80,
                    desc=f"Creating tasks for: {spec_name}",
                    disable=not verbose,
                ):
                    spec_kwargs = self._compute_kwargs(spec, mol_chunk)
                    result = collection.compute(**spec_kwargs)
                    # datasets give a compute response, but opt and torsiondrives give ints
                    try:
                        spec_tasks += result
                    except TypeError:
                        spec_tasks += len(result.ids)

                responses[spec_name] = spec_tasks

        return responses

    @abc.abstractmethod
    def __add__(self, other: "_BaseDataset") -> "_BaseDataset":
        """
        Add two Basicdatasets together.
        """
        raise NotImplementedError()

    @classmethod
    def parse_file(cls, file_name: str):
        """
        Create a Dataset object from a compressed json file.

        Args:
            file_name: The name of the file the dataset should be created from.
        """
        data = deserialize(file_name=file_name)
        return cls(**data)

    def get_molecule_entry(self, molecule: Union[off.Molecule, str]) -> List[str]:
        """
        Search through the dataset for a molecule and return the dataset index of any exact molecule matches.

        Args:
            molecule: The smiles string for the molecule or an openforcefield.topology.Molecule that is to be searched for.

        Returns:
            A list of dataset indices which contain the target molecule.
        """
        # if we have a smiles string convert it
        if isinstance(molecule, str):
            molecule = off.Molecule.from_smiles(molecule, allow_undefined_stereo=True)

        # make a unique inchi key
        inchi_key = molecule.to_inchikey(fixed_hydrogens=False)
        hits = []
        for entry in self.dataset.values():
            if inchi_key == entry.attributes.inchi_key:
                # they have same basic inchi now match the molecule
                if molecule == entry.get_off_molecule(include_conformers=False):
                    hits.append(entry.index)

        return hits

    @property
    def filtered(self) -> off.Molecule:
        """
        A generator which yields a openff molecule representation for each molecule filtered while creating this dataset.

        Note:
            Modifying the molecule will have no effect on the data stored.
        """

        for component, data in self.filtered_molecules.items():
            for smiles in data.molecules:
                offmol = off.Molecule.from_smiles(smiles, allow_undefined_stereo=True)
                yield offmol

    @property
    def n_filtered(self) -> int:
        """
        Calculate the total number of molecules filtered by the components used in a workflow to create this dataset.
        """
        filtered = sum(
            [len(data.molecules) for data in self.filtered_molecules.values()]
        )
        return filtered

    @property
    def n_records(self) -> int:
        """
        Return the total number of records that will be created on submission of the dataset.

        Note:
            * The number returned will be different depending on the dataset used.
            * The amount of unique molecule can be found using `n_molecules`
        """

        n_records = sum([len(data.initial_molecules) for data in self.dataset.values()])
        return n_records

    @property
    def n_molecules(self) -> int:
        """
        Calculate the number of unique molecules to be submitted.

        Notes:
            * This method has been improved for better performance on large datasets and has been tested on an optimization dataset of over 10500 molecules.
            * This function does not calculate the total number of entries of the dataset see `n_records`
        """
        molecules = {}
        for entry in self.dataset.values():
            inchikey = entry.attributes.inchi_key
            try:
                like_mols = molecules[inchikey]
                mol_to_add = entry.get_off_molecule(False).to_inchikey(
                    fixed_hydrogens=True
                )
                for index in like_mols:
                    if mol_to_add == self.dataset[index].get_off_molecule(
                        False
                    ).to_inchikey(fixed_hydrogens=True):
                        break
                else:
                    molecules[inchikey].append(entry.index)
            except KeyError:
                molecules[inchikey] = [
                    entry.index,
                ]
        return sum([len(value) for value in molecules.values()])

    @property
    def molecules(self) -> Generator[off.Molecule, None, None]:
        """
        A generator that creates an openforcefield.topology.Molecule one by one from the dataset.

        Note:
            Editing the molecule will not effect the data stored in the dataset as it is immutable.
        """

        for molecule_data in self.dataset.values():
            # create the molecule from the cmiles data
            yield molecule_data.get_off_molecule(include_conformers=True)

    @property
    def n_components(self) -> int:
        """
        Return the amount of components that have been ran during generating the dataset.
        """

        n_filtered = len(self.filtered_molecules)
        return n_filtered

    @property
    def components(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        """
        Gather the details of the components that were ran during the creation of this dataset.
        """

        components = []
        for component in self.filtered_molecules.values():
            components.append(component.dict(exclude={"molecules"}))

        return components

    def filter_molecules(
        self,
        molecules: Union[off.Molecule, List[off.Molecule]],
        component: str,
        component_settings: Dict[str, Any],
        component_provenance: Dict[str, str],
    ) -> None:
        """
        Filter a molecule or list of molecules by the component they failed.

        Args:
            molecules:
                A molecule or list of molecules to be filtered.
            component_settings:
                The dictionary representation of the component that filtered this set of molecules.
            component:
                The name of the component.
            component_provenance:
                The dictionary representation of the component provenance.
        """

        if isinstance(molecules, off.Molecule):
            # make into a list
            molecules = [molecules]

        if component in self.filtered_molecules:
            filter_mols = [
                molecule.to_smiles(isomeric=True, explicit_hydrogens=True)
                for molecule in molecules
            ]
            self.filtered_molecules[component].molecules.extend(filter_mols)
        else:

            filter_data = FilterEntry(
                off_molecules=molecules,
                component=component,
                component_provenance=component_provenance,
                component_settings=component_settings,
            )

            self.filtered_molecules[filter_data.component] = filter_data

    def add_molecule(
        self,
        index: str,
        molecule: off.Molecule,
        attributes: Union[Dict[str, Any], MoleculeAttributes],
        extras: Optional[Dict[str, Any]] = None,
        keywords: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Add a molecule to the dataset under the given index with the passed cmiles.

        Args:
            index:
                The index that should be associated with the molecule in QCArchive.
            molecule:
                The instance of the molecule which contains its conformer information.
            attributes:
                The attributes dictionary containing all of the relevant identifier tags for the molecule and
                extra meta information on the calculation.
            extras:
                The extras that should be supplied into the qcportal.moldels.Molecule.
            keywords:
                Any extra keywords which are required for the calculation.

        Note:
            Each molecule in this basic dataset should have all of its conformers expanded out into separate entries.
            Thus here we take the general molecule index and increment it.
        """

        try:
            data_entry = self._entry_class()(
                off_molecule=molecule,
                index=index,
                attributes=attributes,
                extras=extras or {},
                keywords=keywords or {},
                **kwargs,
            )
            self.dataset[index] = data_entry
            # add any extra elements to the metadata
            self.metadata.elements.update(data_entry.initial_molecules[0].symbols)

        except qcel.exceptions.ValidationError:
            # the molecule has some qcschema issue and should be removed
            self.filter_molecules(
                molecules=molecule,
                component="QCSchemaIssues",
                component_settings={
                    "component_description": "The molecule was removed as a valid QCSchema could not be made",
                    "type": "QCSchemaIssues",
                },
                component_provenance=self.provenance,
            )

    def _get_missing_basis_coverage(
        self, raise_errors: bool = True
    ) -> Dict[str, Set[str]]:
        """
        Work out if the selected basis set covers all of the elements in the dataset for each specification if not return the missing
        element symbols.

        Args:
            raise_errors: If `True` the function will raise an error for missing basis coverage, else we return the missing data and just print warnings.
        """
        import re
        import warnings

        import basis_set_exchange as bse
        from simtk.openmm.app import Element

        basis_report = {}
        for spec in self.qc_specifications.values():
            if spec.program.lower() == "torchani":
                # check ani1 first
                ani_coverage = {
                    "ani1x": {"C", "H", "N", "O"},
                    "ani1ccx": {"C", "H", "N", "O"},
                    "ani2x": {"C", "H", "N", "O", "S", "F", "Cl"},
                }
                covered_elements = ani_coverage[spec.method.lower()]
                # this is validated at the spec level so we should not get an error here
                difference = self.metadata.elements.difference(covered_elements)

            elif spec.program.lower() == "psi4":
                # now check psi4
                # TODO this list should be updated with more basis transforms as we find them
                psi4_converter = {"dzvp": "dgauss-dzvp"}
                month_subs = {"jun-", "mar-", "apr-", "may-", "feb-"}
                basis = psi4_converter.get(spec.basis.lower(), spec.basis.lower())
                # here we need to apply conversions for special characters to match bse
                # replace the *
                basis = re.sub("\*", "_st_", basis)
                # replace any /
                basis = re.sub("/", "_sl_", basis)
                # check for heavy tags
                basis = re.sub("heavy-", "", basis)
                try:
                    basis_meta = bse.get_metadata()[basis]
                except KeyError:
                    # now try and do month subs
                    for month in month_subs:
                        if month in basis:
                            basis = re.sub(month, "", basis)
                    # now try and get the basis again
                    basis_meta = bse.get_metadata()[basis]

                elements = basis_meta["versions"][basis_meta["latest_version"]][
                    "elements"
                ]
                covered_elements = set(
                    [
                        Element.getByAtomicNumber(int(element)).symbol
                        for element in elements
                    ]
                )
                difference = self.metadata.elements.difference(covered_elements)

            elif spec.program.lower() == "openmm":
                # smirnoff covered elements
                covered_elements = {"C", "H", "N", "O", "P", "S", "Cl", "Br", "F", "I"}
                difference = self.metadata.elements.difference(covered_elements)

            elif spec.program.lower() == "rdkit":
                # all atoms are defined in the uff so return an empty set.
                difference = set()

            else:
                # xtb
                # all atoms are covered and this must be xtb
                difference = set()

            basis_report[spec.spec_name] = difference

        for spec_name, report in basis_report.items():
            if report:
                if raise_errors:
                    raise MissingBasisCoverageError(
                        f"The following elements: {report} are not covered by the selected basis : {self.qc_specifications[spec_name].basis} and method : {self.qc_specifications[spec_name].method}"
                    )
                else:
                    warnings.warn(
                        f"The following elements: {report} are not covered by the selected basis : {self.qc_specifications[spec_name].basis} and method : {self.qc_specifications[spec_name].method}",
                        UserWarning,
                    )
        if not raise_errors:
            return basis_report

    def _get_spec_keywords(self, spec: QCSpec) -> ptl.models.KeywordSet:
        """
        Build a keyword set which is specific to this QC specification and accounts for implicit solvent when requested.
        """
        # use the basic keywords we allow
        data = self.dict(include={"maxiter", "scf_properties"})
        # account for implicit solvent
        if spec.implicit_solvent is not None:
            data["pcm"] = True
            data["pcm__input"] = spec.implicit_solvent.to_string()

        return ptl.models.KeywordSet(values=data)

    def export_dataset(self, file_name: str, compression: Optional[str] = None) -> None:
        """
        Export the dataset to file so that it can be used to make another dataset quickly.

        Args:
            file_name:
                The name of the file the dataset should be wrote to.
            compression:
                The type of compression that should be added to the export.

        Raises:
            UnsupportedFiletypeError: If the requested file type is not supported.


        Note:
            The supported file types are:

            - `json`

            Additionally, the file will automatically compressed depending on the
            final extension if compression is not explicitly supplied:

            - `json.xz`
            - `json.gz`
            - `json.bz2`

            Check serializers.py for more details. Right now bz2 seems to
            produce the smallest files.
        """

        # Check here early, just to filter anything non-json for now
        # Ideally the serializers should be checking this
        split = file_name.split(".")
        split = split[-1:] if len(split) == 1 else split[-2:]
        if "json" not in split:
            raise UnsupportedFiletypeError(
                f"The dataset export file name with leading extension {split[-1]} is not supported, "
                "please end the file name with json."
            )

        serialize(serializable=self, file_name=file_name, compression=compression)

    def coverage_report(self, forcefields: List[str]) -> Dict:
        """
        Produce a coverage report of all of the parameters that are exercised by the molecules in the dataset.

        Args:
            forcefields:
                The name of the openforcefield force field which should be included in the coverage report.

        Returns:
            A dictionary for each of the force fields which break down which parameters are exercised by their
            parameter type.
        """

        from openff.toolkit.typing.engines.smirnoff import ForceField

        coverage = {}
        param_types = {
            "a": "Angles",
            "b": "Bonds",
            "c": "Constraints",
            "t": "ProperTorsions",
            "i": "ImproperTorsions",
            "n": "vdW",
        }
        if isinstance(forcefields, str):
            forcefields = [
                forcefields,
            ]

        for forcefield in forcefields:

            result = {}
            ff = ForceField(forcefield)
            for molecule in self.molecules:
                labels = ff.label_molecules(molecule.to_topology())[0]
                # format the labels into a set
                covered_types = set(
                    [label.id for types in labels.values() for label in types.values()]
                )
                # now insert into the forcefield dict
                for parameter in covered_types:
                    p_id = parameter[0]
                    result.setdefault(param_types[p_id], set()).add(parameter)

            # now store the force field dict into the main result
            coverage[forcefield] = result

        return coverage

    def visualize(
        self,
        file_name: str,
        columns: int = 4,
        toolkit: Optional[Literal["openeye", "rdkit"]] = None,
    ) -> None:
        """
        Create a pdf file of the molecules with any torsions highlighted using either openeye or rdkit.

        Args:
            file_name:
                The name of the pdf file which will be produced.
            columns:
                The number of molecules per row.
            toolkit:
                The option to specify the backend toolkit used to produce the pdf file.
        """

        molecules = []

        for data in self.dataset.values():

            off_mol = data.get_off_molecule(include_conformers=False)
            off_mol.name = None

            if hasattr(data, "dihedrals"):
                off_mol.properties["dihedrals"] = data.dihedrals

            molecules.append(off_mol)

        molecules_to_pdf(molecules, file_name, columns, toolkit)

    def molecules_to_file(self, file_name: str, file_type: str) -> None:
        """
        Write the molecules to the requested file type.

        Args:
            file_name:
                The name of the file the molecules should be stored in.
            file_type:
                The file format that should be used to store the molecules.

        Important:
            The supported file types are:

            - SMI
            - INCHI
            - INCKIKEY
        """

        file_writers = {
            "smi": self._molecules_to_smiles,
            "inchi": self._molecules_to_inchi,
            "inchikey": self._molecules_to_inchikey,
        }

        try:
            # get the list of molecules
            molecules = file_writers[file_type.lower()]()

            with open(file_name, "w") as output:
                for molecule in molecules:
                    output.write(f"{molecule}\n")
        except KeyError:
            raise UnsupportedFiletypeError(
                f"The requested file type {file_type} is not supported, supported types are"
                f"{file_writers.keys()}."
            )

    def _molecules_to_smiles(self) -> List[str]:
        """
        Create a list of molecules canonical isomeric smiles.
        """

        smiles = [
            data.attributes.canonical_isomeric_smiles for data in self.dataset.values()
        ]
        return smiles

    def _molecules_to_inchi(self) -> List[str]:
        """
        Create a list of the molecules standard InChI.
        """

        inchi = [data.attributes.standard_inchi for data in self.dataset.values()]
        return inchi

    def _molecules_to_inchikey(self) -> List[str]:
        """
        Create a list of the molecules standard InChIKey.
        """

        inchikey = [data.attributes.inchi_key for data in self.dataset.values()]
        return inchikey


class BasicDataset(_BaseDataset):
    """
    The general QCFractal dataset class which contains all of the molecules and information about them prior to
    submission.

    The class is a simple holder of the dataset and information about it and can do simple checks on the data before
    submitting it such as ensuring that the molecules have cmiles information
    and a unique index to be identified by.

    Note:
        The molecules in this dataset are all expanded so that different conformers are unique submissions.
    """

    type: Literal["DataSet"] = "DataSet"

    @classmethod
    def _entry_class(cls) -> Type[DatasetEntry]:
        return DatasetEntry

    def __add__(self, other: "BasicDataset") -> "BasicDataset":
        import copy

        # make sure the dataset types match
        if self.type != other.type:
            raise DatasetCombinationError(
                f"The datasets must be the same type, you can not add types {self.type} and {other.type}"
            )

        # create a new datset
        new_dataset = copy.deepcopy(self)
        # update the elements in the dataset
        new_dataset.metadata.elements.update(other.metadata.elements)
        for index, entry in other.dataset.items():
            # search for the molecule
            entry_ids = new_dataset.get_molecule_entry(
                entry.get_off_molecule(include_conformers=False)
            )
            if not entry_ids:
                new_dataset.dataset[index] = entry
            else:
                mol_id = entry_ids[0]
                current_entry = new_dataset.dataset[mol_id]
                _, atom_map = off.Molecule.are_isomorphic(
                    entry.get_off_molecule(include_conformers=False),
                    current_entry.get_off_molecule(include_conformers=False),
                    return_atom_map=True,
                )
                # remap the molecule and all conformers
                entry_mol = entry.get_off_molecule(include_conformers=True)
                mapped_mol = entry_mol.remap(mapping_dict=atom_map, current_to_new=True)
                for i in range(mapped_mol.n_conformers):
                    mapped_schema = mapped_mol.to_qcschema(
                        conformer=i, extras=current_entry.initial_molecules[0].extras
                    )
                    if mapped_schema not in current_entry.initial_molecules:
                        current_entry.initial_molecules.append(mapped_schema)

        return new_dataset

    def _generate_collection(self, client: "FractalClient") -> ptl.collections.Dataset:
        collection = ptl.collections.Dataset(
            name=self.dataset_name,
            client=client,
            default_driver=self.driver,
            default_program="psi4",
            tagline=self.dataset_tagline,
            tags=self.dataset_tags,
            description=self.description,
            provenance=self.provenance,
            metadata=self.metadata.dict(),
        )
        return collection

    def _get_procedure_spec(self):
        """Needed for `submit` usage."""
        return None

    def _get_indices(self, collection: ptl.collections.Dataset):
        return [e.molecule_id for e in collection.data.records]

    def _add_dataset_specification(
        self,
        spec: QCSpec,
        dataset: ptl.collections.Dataset,
        procedure_spec: Optional["OptimizationSpecification"] = None,
    ) -> bool:
        """Add the given compute spec to this Datasets's corresponding Collection.

        Args:
            spec:
                The QCSpec we are trying to add to the dataset in the QCArchive instance.
            dataset:
                The dataset to add this compute spec to.
            procedure_spec:
                The procedure spec to add; ignored for ``BasicDataset``.

        Returns:
            ``True`` if the specification successfully added, ``False`` otherwise.
        """
        # generate the keyword set
        kw = self._get_spec_keywords(spec=spec)
        try:
            # try and add the keywords; if present then continue
            dataset.add_keywords(
                alias=spec.spec_name,
                program=spec.program,
                keyword=kw,
                default=False,
            )
            dataset.save()
            return True
        except (KeyError, AttributeError):
            return False

    def _compute_kwargs(self, spec, indices):
        spec_kwargs = dict(tag=self.compute_tag, priority=self.priority)
        spec_kwargs.update(spec.dict(include={"method", "basis", "program"}))
        spec_kwargs["keywords"] = spec.spec_name
        spec_kwargs["protocols"] = {"wavefunction": spec.store_wavefunction.value}
        spec_kwargs["subset"] = indices
        return spec_kwargs

    def _add_entries(
        self, dataset: ptl.collections.Dataset, chunk_size: int
    ) -> Tuple[List[str], int]:
        new_entries = 0
        indices = []

        for i, (index, data) in enumerate(self.dataset.items()):

            # if we hit the chunk size, we upload to the server
            if (i % chunk_size) == 0:
                dataset.save()

            if len(data.initial_molecules) > 1:

                # check if the index has a number tag
                # if so, start from this tag
                index, tag = self._clean_index(index=index)

                for j, molecule in enumerate(data.initial_molecules):
                    name = index + f"-{tag + j}"
                    new_entries += int(
                        self._add_entry(
                            molecule=molecule, dataset=dataset, name=name, data=data
                        )
                    )
                    indices.append(name)
            else:
                new_entries += int(
                    self._add_entry(
                        molecule=data.initial_molecules[0],
                        dataset=dataset,
                        name=index,
                        data=data,
                    )
                )
                indices.append(index)

        # upload remainder molecules to the server
        dataset.save()
        # we have to convert the indices to the object ids
        object_ids = [
            entry.molecule_id for entry in dataset.data.records if entry.name in indices
        ]

        return object_ids, new_entries

    def _add_entry(
        self,
        molecule: qcel.models.Molecule,
        dataset: ptl.collections.Dataset,
        name: str,
        data: DatasetEntry,
    ) -> bool:
        """
        Attempt to add molecule the dataset.
        Return `True` if successful, `False` otherwise.
        """
        try:
            dataset.add_entry(name=name, molecule=molecule)
            return True
        except KeyError:
            return False


class OptimizationDataset(BasicDataset):
    """
    An optimisation dataset class which handles submission of settings differently from the basic dataset, and creates
    optimization datasets in the public or local qcarcive instance.
    """

    type: Literal["OptimizationDataset"] = "OptimizationDataset"
    driver: DriverEnum = DriverEnum.gradient
    optimization_procedure: GeometricProcedure = Field(
        GeometricProcedure(),
        description="The optimization program and settings that should be used.",
    )
    dataset: Dict[str, OptimizationEntry] = {}

    @classmethod
    def _entry_class(cls) -> Type[OptimizationEntry]:
        return OptimizationEntry

    @validator("driver")
    def _check_driver(cls, driver):
        """Make sure that the driver is set to gradient only and not changed."""
        if driver.value != "gradient":
            driver = DriverEnum.gradient
        return driver

    def __add__(self, other: "OptimizationDataset") -> "OptimizationDataset":
        """
        Add two Optimization datasets together, if the constraints are different then the entries are considered different.
        """
        import copy

        from openff.qcsubmit.utils import remap_list

        # make sure the dataset types match
        if self.type != other.type:
            raise DatasetCombinationError(
                f"The datasets must be the same type, you can not add types {self.type} and {other.type}"
            )

        # create a new dataset
        new_dataset = copy.deepcopy(self)
        # update the elements in the dataset
        new_dataset.metadata.elements.update(other.metadata.elements)
        for entry in other.dataset.values():
            # search for the molecule
            entry_ids = new_dataset.get_molecule_entry(
                entry.get_off_molecule(include_conformers=False)
            )
            if entry_ids:
                records = 0
                for mol_id in entry_ids:
                    current_entry = new_dataset.dataset[mol_id]
                    # for each entry count the number of inputs incase we need a new entry
                    records += len(current_entry.initial_molecules)
                    _, atom_map = off.Molecule.are_isomorphic(
                        entry.get_off_molecule(include_conformers=False),
                        current_entry.get_off_molecule(include_conformers=False),
                        return_atom_map=True,
                    )

                    current_constraints = current_entry.constraints
                    # make sure all constraints are the same
                    # remap the entry to compare
                    entry_constraints = Constraints()
                    for constraint in entry.constraints.freeze:
                        entry_constraints.add_freeze_constraint(
                            constraint.type, remap_list(constraint.indices, atom_map)
                        )
                    for constraint in entry.constraints.set:
                        entry_constraints.add_set_constraint(
                            constraint.type,
                            remap_list(constraint.indices, atom_map),
                            constraint.value,
                        )

                    if current_constraints == entry_constraints:
                        # transfer the entries
                        # remap and transfer
                        off_mol = entry.get_off_molecule(include_conformers=True)
                        mapped_mol = off_mol.remap(
                            mapping_dict=atom_map, current_to_new=True
                        )
                        for i in range(mapped_mol.n_conformers):
                            mapped_schema = mapped_mol.to_qcschema(
                                conformer=i,
                                extras=current_entry.initial_molecules[0].extras,
                            )
                            if mapped_schema not in current_entry.initial_molecules:
                                current_entry.initial_molecules.append(mapped_schema)
                        break
                    # else:
                    #     # if they are not the same move on to the next entry
                    #     continue
                else:
                    # we did not break so add the entry with a new unique index
                    core, tag = self._clean_index(entry.index)
                    entry.index = core + f"-{tag + records}"
                    new_dataset.dataset[entry.index] = entry
            else:
                # if no other molecules just add it
                new_dataset.dataset[entry.index] = entry

        return new_dataset

    def _add_keywords(self, client: "FractalClient", spec: QCSpec) -> str:
        """
        Add the keywords to the client and return the index number of the keyword set.

        Returns:
            kw_id: The keyword index number in the client.
        """

        kw = self._get_spec_keywords(spec=spec)
        kw_id = client.add_keywords([kw])[0]
        return kw_id

    def get_qc_spec(self, spec_name: str, keyword_id: str) -> QCSpecification:
        """
        Create the QC specification for the computation.

        Args:
            spec_name: The name of the spec we want to convert to a QCSpecification
            keyword_id: The string of the keyword set id number.

        Returns:
            The dictionary representation of the QC specification
        """
        spec = self.qc_specifications[spec_name]
        qc_spec = QCSpecification(
            driver=self.driver,
            method=spec.method,
            basis=spec.basis,
            keywords=keyword_id,
            program=spec.program,
            protocols={"wavefunction": spec.store_wavefunction},
        )

        return qc_spec

    def _add_dataset_specification(
        self,
        spec: QCSpec,
        dataset: ptl.collections.OptimizationDataset,
        procedure_spec: Optional["OptimizationSpecification"] = None,
    ) -> bool:
        """Add the given compute spec to this Datasets's corresponding Collection.

        Args:
            spec:
                The QCSpec we are trying to add to the collection
            dataset:
                The QCArchive dataset this specification should be added to
            procedure_spec:
                The qcportal style optimization spec

        Raises:
            QCSpecificationError: If a specification with the same name is already added to the collection but has different settings.

        Note:
            If a specification is already stored under this name in the collection we have options:
            - If a spec with the same name but different details has been added and used we must raise an error to change the name of the new spec
            - If the spec has been added and has not been used then overwrite it.
        """
        # build the qcportal version of our spec
        kw_id = self._add_keywords(client=dataset.client, spec=spec)
        qcportal_spec = self.get_qc_spec(spec_name=spec.spec_name, keyword_id=kw_id)

        # see if the spec is in the history
        if spec.spec_name.lower() in dataset.data.history:
            collection_spec = dataset.get_specification(name=spec.spec_name)
            # check they are the same
            if (
                collection_spec.optimization_spec == procedure_spec
                and qcportal_spec == collection_spec.qc_spec
            ):
                # the spec is already there and is the same so just skip adding it
                return True
            else:
                raise QCSpecificationError(
                    f"A specification with the name {spec.spec_name} is already registered with the collection but has different settings and has already been used and should not be overwriten. "
                    f"Please change the name of this specification to continue."
                )

        else:
            # the spec either has not been added or has not been used so set the new default
            dataset.add_specification(
                name=spec.spec_name,
                optimization_spec=procedure_spec,
                qc_spec=qcportal_spec,
                description=spec.spec_description,
                overwrite=True,
            )
            return True

    def _generate_collection(
        self, client: "FractalClient"
    ) -> ptl.collections.OptimizationDataset:
        collection = ptl.collections.OptimizationDataset(
            name=self.dataset_name,
            client=client,
            tagline=self.dataset_tagline,
            tags=self.dataset_tags,
            description=self.description,
            provenance=self.provenance,
            metadata=self.metadata.dict(),
        )
        return collection

    def _get_procedure_spec(self):
        return self.optimization_procedure.get_optimzation_spec()

    def _get_indices(self, collection):
        return collection.df.index.tolist()

    def _compute_kwargs(self, spec, indices):
        spec_kwargs = dict(tag=self.compute_tag, priority=self.priority)
        spec_kwargs["subset"] = indices
        spec_kwargs["specification"] = spec.spec_name
        return spec_kwargs

    def _add_entries(
        self, dataset: ptl.collections.OptimizationDataset, chunk_size: int
    ) -> Tuple[List[str], int]:
        new_entries = 0
        indices = []

        for i, (index, data) in enumerate(self.dataset.items()):

            # if we hit the chunk size, we upload to the server
            if (i % chunk_size) == 0:
                dataset.save()

            if len(data.initial_molecules) > 1:

                # check if the index has a number tag
                # if so, start from this tag
                index, tag = self._clean_index(index=index)

                for j, molecule in enumerate(data.initial_molecules):
                    name = index + f"-{tag + j}"
                    new_entries += int(
                        self._add_entry(
                            molecule=molecule, dataset=dataset, name=name, data=data
                        )
                    )
                    indices.append(name)
            else:
                new_entries += int(
                    self._add_entry(
                        molecule=data.initial_molecules[0],
                        dataset=dataset,
                        name=index,
                        data=data,
                    )
                )
                indices.append(index)

        # upload remainder molecules to the server
        dataset.save()

        return indices, new_entries

    def _add_entry(
        self,
        molecule: qcel.models.Molecule,
        dataset: ptl.collections.OptimizationDataset,
        name: str,
        data: OptimizationEntry,
    ) -> bool:
        """
        Add a molecule to the given optimization dataset and return the ids and the result of adding the molecule.
        """
        try:
            dataset.add_entry(
                name=name,
                initial_molecule=molecule,
                attributes=data.attributes.dict(),
                additional_keywords=data.formatted_keywords,
                save=False,
            )
            return True
        except KeyError:
            return False


class TorsiondriveDataset(OptimizationDataset):
    """
    An torsiondrive dataset class which handles submission of settings differently from the basic dataset, and creates
    torsiondrive datasets in the public or local qcarcive instance.

    Important:
        The dihedral_ranges for the whole dataset can be defined here or if different scan ranges are required on a case
        by case basis they can be defined for each torsion in a molecule separately in the keywords of the torsiondrive entry.
    """

    dataset: Dict[str, TorsionDriveEntry] = {}
    type: Literal["TorsionDriveDataset"] = "TorsionDriveDataset"
    optimization_procedure: GeometricProcedure = GeometricProcedure.parse_obj(
        {"enforce": 0.1, "reset": True, "qccnv": True, "epsilon": 0.0}
    )
    grid_spacing: List[int] = Field(
        [15],
        description="The grid spcaing that should be used for all torsiondrives, this can be overwriten on a per entry basis.",
    )
    energy_upper_limit: float = Field(
        0.05,
        description="The upper energy limit to spawn new optimizations in the torsiondrive.",
    )
    dihedral_ranges: Optional[List[Tuple[int, int]]] = Field(
        None,
        description="The scan range that should be used for each torsiondrive, this can be overwriten on a per entry basis.",
    )
    energy_decrease_thresh: Optional[float] = Field(
        None,
        description="The energy lower threshold to trigger new optimizations in the torsiondrive.",
    )

    @classmethod
    def _entry_class(cls) -> Type[TorsionDriveEntry]:
        return TorsionDriveEntry

    def __add__(self, other: "TorsiondriveDataset") -> "TorsiondriveDataset":
        """
        Add two TorsiondriveDatasets together, if the central bond in the dihedral is the same the entries are considered the same.
        """
        import copy

        # make sure the dataset types match
        if self.type != other.type:
            raise DatasetCombinationError(
                f"The datasets must be the same type, you can not add types {self.type} and {other.type}"
            )

        # create a new dataset
        new_dataset = copy.deepcopy(self)
        # update the elements in the dataset
        new_dataset.metadata.elements.update(other.metadata.elements)
        for index, entry in other.dataset.items():
            # search for the molecule
            entry_ids = new_dataset.get_molecule_entry(
                entry.get_off_molecule(include_conformers=False)
            )
            for mol_id in entry_ids:
                current_entry = new_dataset.dataset[mol_id]
                _, atom_map = off.Molecule.are_isomorphic(
                    entry.get_off_molecule(include_conformers=False),
                    current_entry.get_off_molecule(include_conformers=False),
                    return_atom_map=True,
                )

                # gather the current dihedrals forward and backwards
                current_dihedrals = set(
                    [(dihedral[1:3]) for dihedral in current_entry.dihedrals]
                )
                for dihedral in current_entry.dihedrals:
                    current_dihedrals.add((dihedral[1:3]))
                    current_dihedrals.add((dihedral[2:0:-1]))

                # now gather the other entry dihedrals forwards and backwards
                other_dihedrals = set()
                for dihedral in entry.dihedrals:
                    other_dihedrals.add(tuple(atom_map[i] for i in dihedral[1:3]))
                    other_dihedrals.add(tuple(atom_map[i] for i in dihedral[2:0:-1]))

                difference = current_dihedrals - other_dihedrals
                if not difference:
                    # the entry is already there so add new conformers and skip
                    off_mol = entry.get_off_molecule(include_conformers=True)
                    mapped_mol = off_mol.remap(
                        mapping_dict=atom_map, current_to_new=True
                    )
                    for i in range(mapped_mol.n_conformers):
                        mapped_schema = mapped_mol.to_qcschema(
                            conformer=i,
                            extras=current_entry.initial_molecules[0].extras,
                        )
                        if mapped_schema not in current_entry.initial_molecules:
                            current_entry.initial_molecules.append(mapped_schema)
                    break
            else:
                # none of the entries matched so add it
                new_dataset.dataset[index] = entry

        return new_dataset

    @property
    def n_records(self) -> int:
        """
        Calculate the number of records that will be submitted.
        """
        return len(self.dataset)

    def _generate_collection(
        self, client: "FractalClient"
    ) -> ptl.collections.TorsionDriveDataset:
        collection = ptl.collections.TorsionDriveDataset(
            name=self.dataset_name,
            client=client,
            tagline=self.dataset_tagline,
            tags=self.dataset_tags,
            description=self.description,
            provenance=self.provenance,
            metadata=self.metadata.dict(),
        )
        return collection

    def _add_entries(
        self, dataset: ptl.collections.TorsionDriveDataset, chunk_size: int
    ) -> Tuple[List[str], int]:
        new_entries = 0
        indices = []

        for i, (index, data) in enumerate(self.dataset.items()):
            # if we hit the chunk size, we upload to the server
            if (i % chunk_size) == 0:
                dataset.save()

            new_entries += int(
                self._add_entry(
                    data.initial_molecules, dataset=dataset, name=data.index, data=data
                )
            )
            indices.append(index)

        # upload remainder molecules to the server
        dataset.save()

        return indices, new_entries

    def _add_entry(
        self,
        molecule: qcel.models.Molecule,
        dataset: ptl.collections.TorsionDriveDataset,
        name: str,
        data: TorsionDriveEntry,
    ) -> bool:
        """
        Add a molecule to the given torsiondrive dataset and return the id and
        the result of adding the molecule."""
        try:
            dataset.add_entry(
                name=data.index,
                initial_molecules=data.initial_molecules,
                dihedrals=data.dihedrals,
                grid_spacing=data.keywords.grid_spacing or self.grid_spacing,
                energy_upper_limit=data.keywords.energy_upper_limit
                or self.energy_upper_limit,
                attributes=data.attributes.dict(),
                energy_decrease_thresh=data.keywords.energy_decrease_thresh
                or self.energy_decrease_thresh,
                dihedral_ranges=data.keywords.dihedral_ranges or self.dihedral_ranges,
            )
            return True
        except KeyError:
            return False
