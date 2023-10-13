import abc
import json
from collections import defaultdict
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

import qcelemental as qcel
import qcportal as ptl
from openff.toolkit import topology as off
from pydantic import Field, constr, validator
from qcelemental.models import AtomicInput, OptimizationInput
from qcelemental.models.procedures import QCInputSpecification
from qcportal import PortalClient, PortalRequestError
from qcportal.optimization import OptimizationDatasetNewEntry, OptimizationSpecification
from qcportal.singlepoint import (
    QCSpecification,
    SinglepointDatasetNewEntry,
    SinglepointDriver,
)
from qcportal.torsiondrive import TorsiondriveDatasetNewEntry, TorsiondriveSpecification
from typing_extensions import Literal

from openff.qcsubmit.common_structures import CommonBase, Metadata, MoleculeAttributes
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
    UnsupportedFiletypeError,
)
from openff.qcsubmit.procedures import GeometricProcedure
from openff.qcsubmit.serializers import deserialize, serialize
from openff.qcsubmit.utils.smirnoff import smirnoff_coverage
from openff.qcsubmit.utils.visualize import molecules_to_pdf

if TYPE_CHECKING:
    from openff.toolkit.typing.engines.smirnoff import ForceField

C = TypeVar("C", bound="Collection")
E = TypeVar("E", bound=DatasetEntry)


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
        description="The tagline should be a short description of the dataset which will be displayed by the QCArchive client when the datasets are listed.",
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
    def _generate_collection(self, client: "PortalClient") -> C:
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
    def _get_specifications(self) -> "OptimizationSpecification":
        """Get the procedure spec, if applicable, for this Dataset.

        If the dataset has no concept of procedure specs, this method
        should return `None`.

        Returns:
            Specification for the optimization procedure to perform.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_entries(self) -> List[Any]:
        """Add entries to the Dataset's corresponding Collection.

        This method allows for handling of e.g. generating the index/name for
        the corresponding Collection from each item in `self.dataset`. Since
        each item may feature more than one conformer, appropriate handling
        differs between e.g. `OptimizationDataset` and `TorsiondriveDataset`

        Returns
        -------
        :
            List of new entries to add to the dataset on the server
        """
        pass

    @abc.abstractmethod
    def to_tasks(self) -> Dict[str, List[Union[AtomicInput, OptimizationInput]]]:
        """
        Create a dictionary of QCengine tasks which correspond to this dataset stored by the program which should be used for the task.
        """
        raise NotImplementedError()

    def submit(
        self,
        client: "PortalClient",
        ignore_errors: bool = False,
        find_existing: bool = True,
    ) -> Dict:
        """
        Submit the dataset to a QCFractal server.

        Args:
            client:
                Instance of a portal client
            ignore_errors:
                If the user wants to submit the compute regardless of errors set this to ``True``.
                Mainly to override basis coverage.

        Returns:
            A dictionary of the compute response from the client for each specification submitted.

        Raises:
            MissingBasisCoverageError:
                If the chosen basis set does not cover some of the elements in the dataset.

        """
        from openff.qcsubmit.datasets import legacy_qcsubmit_ds_type_to_next_qcf_ds_type

        # pre submission checks
        # make sure we have some QCSpec to submit
        self._check_qc_specs()
        # basis set coverage check
        self._get_missing_basis_coverage(raise_errors=(not ignore_errors))

        # see if collection already exists
        # if so, we'll extend it
        # if not, we'll create a new one

        try:
            # collection = client.get_dataset(self.type, self.dataset_name)
            qcf_ds_type = legacy_qcsubmit_ds_type_to_next_qcf_ds_type[self.type]
            collection = client.get_dataset(qcf_ds_type, self.dataset_name)
        except PortalRequestError as e:
            self.metadata.validate_metadata(raise_errors=not ignore_errors)
            collection = self._generate_collection(client=client)

        # create specifications
        # TODO - check if specifications already exist
        specs = self._get_specifications()
        for spec_name, spec in specs.items():
            # Send the new specifications to the server
            collection.add_specification(name=spec_name, specification=spec)

        # add the molecules/entries to the database
        entries = self._get_entries()

        # TODO - check if entries already exist
        collection.add_entries(entries)

        return collection.submit(
            tag=self.compute_tag,
            priority=self.priority,
            # find_existing=find_existing
        )

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
        molecule: Optional[off.Molecule],
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
            extras:
                The extras that should be supplied into the qcportal.moldels.Molecule.
            keywords:
                Any extra keywords which are required for the calculation.

        Note:
            Each molecule in this basic dataset should have all of its conformers expanded out into separate entries.
            Thus here we take the general molecule index and increment it.
        """
        # only use attributes if supplied else generate
        # Note we should only reuse attributes if making a dataset from a result so the attributes are consistent
        if "attributes" in kwargs:
            attributes = kwargs.pop("attributes")
        else:
            attributes = MoleculeAttributes.from_openff_molecule(molecule=molecule)

        try:
            data_entry = self._entry_class()(
                off_molecule=molecule,
                attributes=attributes,
                index=index,
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

        try:
            from openmm.app import Element
        except ImportError:
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
                if spec.basis is not None:
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
                else:
                    # the basis is wrote with the method so print a warning about validation
                    warnings.warn(
                        f"The spec {spec.spec_name} has a basis of None, this will not be validated.",
                        UserWarning,
                    )
                    difference = set()

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

    def coverage_report(
        self, force_field: "ForceField", verbose: bool = False
    ) -> Dict[str, Dict[str, int]]:
        """Returns a summary of how many molecules within this dataset would be assigned
        each of the parameters in a force field.

        Notes:
            * Parameters which would not be assigned to any molecules in the dataset
              will not be included in the returned summary.

        Args:
            force_field: The force field containing the parameters to summarize.
            verbose: If true a progress bar will be shown on screen.

        Returns:
            A dictionary of the form ``coverage[handler_name][parameter_smirks] = count``
            which stores the number of molecules within this dataset that would be
            assigned to each parameter.
        """

        return smirnoff_coverage(self.molecules, force_field, verbose)

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


# TODO: SinglepointDataset
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

    def _generate_collection(
        self, client: "PortalClient"
    ) -> ptl.singlepoint.SinglepointDataset:
        return client.add_dataset(
            dataset_type="singlepoint",
            name=self.dataset_name,
            tagline=self.dataset_tagline,
            tags=self.dataset_tags,
            description=self.description,
            provenance=self.provenance,
            default_tag=self.compute_tag,
            default_priority=self.priority,
            metadata=self.metadata.dict(),
        )

    def _get_specifications(self) -> Dict[str, QCSpecification]:
        """Needed for `submit` usage."""

        ret = {}
        for spec_name, spec in self.qc_specifications.items():
            ret[spec_name] = QCSpecification(
                driver=self.driver,
                method=spec.method,
                basis=spec.basis,
                keywords=spec.keywords,
                program=spec.program,
                protocols={"wavefunction": spec.store_wavefunction},
            )

        return ret

    def _get_entries(self) -> List[SinglepointDatasetNewEntry]:
        entries: List[SinglepointDatasetNewEntry] = []

        for entry_name, entry in self.dataset.items():
            if len(entry.initial_molecules) > 1:
                # check if the index has a number tag
                # if so, start from this tag
                index, tag = self._clean_index(index=entry_name)

                for j, molecule in enumerate(entry.initial_molecules):
                    name = index + f"-{tag + j}"
                    entries.append(
                        SinglepointDatasetNewEntry(name=name, molecule=molecule)
                    )
            else:
                entries.append(
                    SinglepointDatasetNewEntry(
                        name=entry_name,
                        molecule=entry.initial_molecules[0],
                    )
                )

        return entries

    def to_tasks(self) -> Dict[str, List[AtomicInput]]:
        """
        Build a dictionary of single QCEngine tasks that correspond to this dataset organised by program name. The tasks can be passed directly
        to qcengine.compute.
        """
        data = defaultdict(list)
        for spec in self.qc_specifications.values():
            qc_model = spec.qc_model
            keywords = spec.qc_keywords
            protocols = {"wavefunction": spec.store_wavefunction.value}
            program = spec.program.lower()
            for index, entry in self.dataset.items():
                # check if the index has a number tag
                # if so, start from this tag
                index, tag = self._clean_index(index=index)
                for j, molecule in enumerate(entry.initial_molecules):
                    name = index + f"-{tag + j}"

                    data[program].append(
                        AtomicInput(
                            id=name,
                            molecule=molecule,
                            driver=self.driver,
                            model=qc_model,
                            keywords=keywords,
                            protocols=protocols,
                        )
                    )
        return data


class OptimizationDataset(BasicDataset):
    """
    An optimisation dataset class which handles submission of settings differently from the basic dataset, and creates
    optimization datasets in the public or local qcarchive instance.
    """

    type: Literal["OptimizationDataset"] = "OptimizationDataset"
    driver: SinglepointDriver = SinglepointDriver.deferred
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
        """Make sure that the driver is set to deferred only and not changed."""
        return SinglepointDriver.deferred

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

    def _generate_collection(
        self, client: "PortalClient"
    ) -> ptl.optimization.OptimizationDataset:
        return client.add_dataset(
            dataset_type="optimization",
            name=self.dataset_name,
            tagline=self.dataset_tagline,
            tags=self.dataset_tags,
            description=self.description,
            provenance=self.provenance,
            default_tag=self.compute_tag,
            default_priority=self.priority,
            metadata=self.metadata.dict(),
        )

    def _get_specifications(self) -> Dict[str, OptimizationSpecification]:
        opt_kw = self.optimization_procedure.get_optimzation_keywords()

        ret = {}

        for spec_name, spec in self.qc_specifications.items():
            qc_spec = QCSpecification(
                driver=self.driver,
                method=spec.method,
                basis=spec.basis,
                keywords=spec.keywords,
                program=spec.program,
                protocols={"wavefunction": spec.store_wavefunction},
            )

            ret[spec_name] = OptimizationSpecification(
                program=self.optimization_procedure.program,
                qc_specification=qc_spec,
                keywords=opt_kw,
            )

        return ret

    def _get_entries(self) -> List[OptimizationDatasetNewEntry]:
        entries: List[OptimizationDatasetNewEntry] = []

        for entry_name, entry in self.dataset.items():
            # TODO this probably needs even more keywords
            opt_kw = dict(constraints=entry.constraints)
            opt_kw.update(entry.keywords)
            if len(entry.initial_molecules) > 1:
                # check if the index has a number tag
                # if so, start from this tag
                index, tag = self._clean_index(index=entry_name)

                for j, molecule in enumerate(entry.initial_molecules):
                    name = index + f"-{tag + j}"
                    entries.append(
                        OptimizationDatasetNewEntry(
                            name=name,
                            initial_molecule=molecule,
                            additional_keywords=opt_kw,
                        )
                    )
            else:
                entries.append(
                    OptimizationDatasetNewEntry(
                        name=entry_name,
                        initial_molecule=entry.initial_molecules[0],
                        additional_keywords=opt_kw,
                    )
                )

        return entries

    def to_tasks(self) -> Dict[str, List[OptimizationInput]]:
        """
        Build a list of QCEngine optimisation inputs organised by the optimisation engine which should be used to run the task.
        """
        data = defaultdict(list)
        opt_program = self.optimization_procedure.program.lower()
        for spec in self.qc_specifications.values():
            qc_model = spec.qc_model
            qc_keywords = spec.qc_keywords
            qc_spec = QCInputSpecification(
                driver=self.driver, model=qc_model, keywords=qc_keywords
            )
            opt_spec = self.optimization_procedure.dict(exclude={"program"})
            # this needs to be the single point calculation program
            opt_spec["program"] = spec.program.lower()

            for index, entry in self.dataset.items():
                index, tag = self._clean_index(index=index)
                for j, molecule in enumerate(entry.initial_molecules):
                    name = index + f"-{tag + j}"

                    data[opt_program].append(
                        OptimizationInput(
                            id=name,
                            keywords=opt_spec,
                            input_specification=qc_spec,
                            initial_molecule=molecule,
                        )
                    )

        return data


class TorsiondriveDataset(OptimizationDataset):
    """
    An torsiondrive dataset class which handles submission of settings differently from the basic dataset, and creates
    torsiondrive datasets in the public or local qcarchive instance.

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
        self, client: "PortalClient"
    ) -> ptl.torsiondrive.TorsiondriveDataset:
        return client.add_dataset(
            dataset_type="torsiondrive",
            name=self.dataset_name,
            tagline=self.dataset_tagline,
            tags=self.dataset_tags,
            description=self.description,
            provenance=self.provenance,
            default_tag=self.compute_tag,
            default_priority=self.priority,
            metadata=self.metadata.dict(),
        )

    def _get_specifications(self) -> Dict[str, TorsiondriveSpecification]:
        td_kw = dict(
            grid_spacing=self.grid_spacing,
            dihedral_ranges=self.dihedral_ranges,
            energy_decrease_thresh=self.energy_decrease_thresh,
            energy_upper_limit=self.energy_upper_limit,
        )

        ret = {}
        for spec_name, spec in self.qc_specifications.items():
            qc_spec = QCSpecification(
                driver=self.driver,
                method=spec.method,
                basis=spec.basis,
                keywords=spec.keywords,
                program=spec.program,
                protocols={"wavefunction": spec.store_wavefunction},
            )
            spec = OptimizationSpecification(
                program=self.optimization_procedure.program,
                qc_specification=qc_spec,
            )
            ret[spec_name] = TorsiondriveSpecification(
                optimization_specification=spec,
                keywords=td_kw,
            )

        return ret

    def _get_entries(self) -> List[TorsiondriveDatasetNewEntry]:
        entries: List[TorsiondriveDatasetNewEntry] = []

        for entry_name, entry in self.dataset.items():
            td_keywords = dict(
                grid_spacing=self.grid_spacing,
                energy_upper_limit=self.energy_upper_limit,
                energy_decrease_thresh=self.energy_decrease_thresh,
                dihedral_ranges=self.dihedral_ranges,
                dihedrals=entry.dihedrals,
            )

            td_keywords.update(entry.keywords.dict(exclude_defaults=True))

            entries.append(
                TorsiondriveDatasetNewEntry(
                    name=entry_name,
                    initial_molecules=entry.initial_molecules,
                    additional_keywords=td_keywords,
                )
            )

        return entries

    def to_tasks(self) -> Dict[str, List[OptimizationInput]]:
        """Build a list of QCEngine procedure tasks which correspond to this dataset."""

        raise NotImplementedError()
