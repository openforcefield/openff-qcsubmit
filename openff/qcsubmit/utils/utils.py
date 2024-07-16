import logging
import os
from contextlib import contextmanager
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from openff.toolkit import topology as off
from openff.toolkit.utils.toolkits import (
    RDKitToolkitWrapper,
    UndefinedStereochemistryError,
)
from qcportal import PortalClient
from qcportal.cache import RecordCache, get_records_with_cache
from qcportal.optimization.record_models import OptimizationRecord
from qcportal.singlepoint.record_models import SinglepointRecord
from qcportal.torsiondrive.record_models import TorsiondriveRecord

logger = logging.getLogger(__name__)


class CachedPortalClient(PortalClient):
    """A cached version of a `qcportal.PortalClient
    <https://molssi.github.io/QCFractal/user_guide/qcportal_reference.html#qcportal.client.PortalClient>`_.
    """

    def __init__(
        self,
        address: str,
        cache_dir: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify: bool = True,
        show_motd: bool = True,
        *,
        cache_max_size: int = 0,
        memory_cache_key: Optional[str] = None,
    ):
        """Parameters
        ----------
        address
            The host or IP address of the FractalServer instance, including
            protocol and port if necessary ("https://ml.qcarchive.molssi.org",
            "http://192.168.1.10:8888")
        cache_dir
            Directory to store an internal cache of records and other data.
            Unlike a normal ``PortalClient``, this argument is required.
        username
            The username to authenticate with.
        password
            The password to authenticate with.
        verify
            Verifies the SSL connection with a third party server. This may be
            False if a FractalServer was not provided an SSL certificate and
            defaults back to self-signed SSL keys.
        show_motd
            If a Message-of-the-Day is available, display it
        cache_max_size
            Maximum size of the cache directory
        """
        super().__init__(
            address,
            username=username,
            password=password,
            verify=verify,
            show_motd=show_motd,
            cache_dir=cache_dir,
            cache_max_size=cache_max_size,
            memory_cache_key=memory_cache_key,
        )
        self.record_cache = RecordCache(
            os.path.join(self.cache.cache_dir, "cache.sqlite"), read_only=False
        )

    def __repr__(self) -> str:
        """A short representation of the current PortalClient.

        Returns
        -------
        str
            The desired representation.
        """
        ret = "CachedPortalClient(server_name='{}', address='{}', username='{}', cache_dir='{}')".format(
            self.server_name, self.address, self.username, self.cache.cache_dir
        )
        return ret

    def get_optimizations(
        self,
        record_ids: Union[int, Sequence[int]],
        missing_ok: bool = False,
        *,
        include: Optional[Iterable[str]] = None,
    ) -> Union[Optional[OptimizationRecord], List[Optional[OptimizationRecord]]]:
        """Obtain optimization records with the specified IDs.

        Records will be returned in the same order as the record ids.

        Parameters
        ----------
        record_ids
            Single ID or sequence/list of records to obtain
        missing_ok
            Unlike a ``PortalClient``, this argument is ignored. If set to
            True, a warning will be printed. Any missing records will cause a
            ``RuntimeError`` to be raised.
        include
            Additional fields to include in the returned record

        Returns
        -------
        :
            If a single ID was specified, returns just that record. Otherwise,
            returns a list of records.
        """
        if missing_ok:
            logger.warning("missing_ok provided but unused by CachedPortalClient")
        if unpack := not isinstance(record_ids, Sequence):
            record_ids = [record_ids]
        res = get_records_with_cache(
            client=self,
            record_cache=self.record_cache,
            record_type=OptimizationRecord,
            record_ids=record_ids,
            include=include,
            force_fetch=False,
        )
        if unpack:
            return res[0]
        else:
            return res

    def get_singlepoints(
        self,
        record_ids: Union[int, Sequence[int]],
        missing_ok: bool = False,
        *,
        include: Optional[Iterable[str]] = None,
    ) -> Union[Optional[SinglepointRecord], List[Optional[SinglepointRecord]]]:
        """
        Obtain singlepoint records with the specified IDs.

        Records will be returned in the same order as the record ids.

        Parameters
        ----------
        record_ids
            Single ID or sequence/list of records to obtain
        missing_ok
            Unlike a ``PortalClient``, this argument is ignored. If set to
            True, a warning will be printed. Any missing records will cause a
            ``RuntimeError`` to be raised.
        include
            Additional fields to include in the returned record

        Returns
        -------
        :
            If a single ID was specified, returns just that record. Otherwise,
            returns a list of records.
        """
        if missing_ok:
            logger.warning("missing_ok provided but unused by CachedPortalClient")
        if unpack := not isinstance(record_ids, Sequence):
            record_ids = [record_ids]
        res = get_records_with_cache(
            client=self,
            record_cache=self.record_cache,
            record_type=SinglepointRecord,
            record_ids=record_ids,
            include=include,
            force_fetch=False,
        )
        if unpack:
            return res[0]
        else:
            return res

    def get_torsiondrives(
        self,
        record_ids: Union[int, Sequence[int]],
        missing_ok: bool = False,
        *,
        include: Optional[Iterable[str]] = None,
    ) -> Union[Optional[TorsiondriveRecord], List[Optional[TorsiondriveRecord]]]:
        """
        Obtain torsiondrive records with the specified IDs.

        Records will be returned in the same order as the record ids.

        Parameters
        ----------
        record_ids
            Single ID or sequence/list of records to obtain
        missing_ok
            Unlike a ``PortalClient``, this argument is ignored. If set to
            True, a warning will be printed. Any missing records will cause a
            ``RuntimeError`` to be raised.
        include
            Additional fields to include in the returned record

        Returns
        -------
        :
            If a single ID was specified, returns just that record. Otherwise,
            returns a list of records.
        """
        if missing_ok:
            logger.warning("missing_ok provided but unused by CachedPortalClient")
        if unpack := not isinstance(record_ids, Sequence):
            record_ids = [record_ids]
        res = get_records_with_cache(
            client=self,
            record_cache=self.record_cache,
            record_type=TorsiondriveRecord,
            record_ids=record_ids,
            include=include,
            force_fetch=False,
        )
        if unpack:
            return res[0]
        else:
            return res

    @contextmanager
    def _no_session(self):
        """This is a supplemental context manager to the ``no_internet``
        manager in _tests/utils/test_manager.py. ``PortalClient`` creates a
        ``requests.Session`` on initialization that can be reused without
        accessing ``socket.socket`` again. Combining ``no_internet`` and
        ``client._no_session`` should completely ensure that the local cache is
        used rather than re-fetching data from QCArchive.
        """
        tmp = self._req_session
        self._req_session = None
        try:
            yield
        finally:
            self._req_session = tmp


def _default_portal_client(client_address) -> PortalClient:
    return PortalClient(client_address)


@contextmanager
def portal_client_manager(portal_client_fn: Callable[[str], PortalClient]):
    """A context manager that temporarily changes the default
    ``qcportal.PortalClient`` constructor used internally in functions like
    ``BasicResultCollection.to_records`` and many of the ``ResultFilter``
    classes. This can be especially useful if you need to provide additional
    keyword arguments to the ``PortalClient``, such as ``verify=False`` or a
    ``cache_dir``.

    Parameters
    ----------
    portal_client_fn:
        A function returning a PortalClient

    Examples
    --------

    Assuming you already have a dataset defined as ``ds``, call ``to_records``
    and use an existing cache in the current working directory if present or
    create a new one automatically:

    >>> from openff.qcsubmit.utils import portal_client_manager
    >>> from qcportal import PortalClient
    >>> def my_portal_client(client_address):
    >>>     return PortalClient(client_address, cache_dir=".")
    >>> with portal_client_manager(my_portal_client):
    >>>     records_and_molecules = ds.to_records()

    """
    global _default_portal_client
    original_client_fn = _default_portal_client
    _default_portal_client = portal_client_fn
    try:
        yield
    finally:
        _default_portal_client = original_client_fn


def get_data(relative_path):
    """
    Get the file path to some data in the qcsubmit package.

    Parameters:
        relative_path: The relative path to the data
    """

    import os

    from pkg_resources import resource_filename

    fn = resource_filename("openff.qcsubmit", os.path.join("data", relative_path))

    if not os.path.exists(fn):
        raise ValueError(
            f"Sorry! {fn} does not exist. If you just added it, you'll have to re-install"
        )

    return fn


def check_missing_stereo(molecule: off.Molecule) -> bool:
    """
    Get if the given molecule has missing stereo by round trip and catching stereo errors.
    Here we use the RDKit backend explicitly for this check as this avoids nitrogen stereochemistry issues with the toolkit.

    Parameters
    ----------
    molecule: off.Molecule
        The molecule which should be checked for stereo issues.

    Returns
    -------
    bool
        `True` if some stereochemistry is missing else `False`.
    """
    try:
        _ = off.Molecule.from_smiles(
            smiles=molecule.to_smiles(isomeric=True, explicit_hydrogens=True),
            hydrogens_are_explicit=True,
            allow_undefined_stereo=False,
            toolkit_registry=RDKitToolkitWrapper(),
        )
        return False
    except UndefinedStereochemistryError:
        return True


def clean_strings(string_list: List[str]) -> List[str]:
    """
    Clean up a list of strings ready to be cast to numbers.
    """
    clean_string = []
    for string in string_list:
        new_string = string.strip()
        clean_string.append(new_string.strip(","))
    return clean_string


def remap_list(target_list: List[int], mapping: Dict[int, int]) -> List[int]:
    """
    Take a list of atom indices and remap them using the given mapping.
    """
    return [mapping[x] for x in target_list]


def condense_molecules(molecules: List[off.Molecule]) -> off.Molecule:
    """
    Take a list of identical molecules in different conformers and collapse them making sure that they are in the same order.
    """
    molecule = molecules.pop()
    for conformer in molecules:
        _, atom_map = off.Molecule.are_isomorphic(
            conformer, molecule, return_atom_map=True
        )
        mapped_mol = conformer.remap(atom_map)
        for geometry in mapped_mol.conformers:
            molecule.add_conformer(geometry)
    return molecule


def chunk_generator(iterable: List, chunk_size: int) -> Generator[List, None, None]:
    """
    Take an iterable and return a list of lists of the specified size.

    Parameters:
         iterable: An iterable object like a list
         chunk_size: The size of each chunk
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def get_torsion(bond: off.Bond) -> Tuple[int, int, int, int]:
    """
    Create a torsion tuple which will be restrained in the torsiondrive.

    Parameters:
        bond: The tuple of the atom indexes for the central bond.

    Returns:
        The tuple of the four atom indices which should be restrained.

    Note:
        If there is more than one possible combination of atoms the heaviest set are selected to be restrained.
    """

    atoms: List[off.Atom] = [bond.atom1, bond.atom2]
    terminal_atoms: Dict[off.Atom, off.atom] = dict()

    for atom in atoms:
        for neighbour in atom.bonded_atoms:
            if neighbour not in atoms:
                # If we have not seen any possible terminal atoms for this atom, add the neighbour
                if atom not in terminal_atoms:
                    terminal_atoms[atom] = neighbour
                # If the neighbour is heavier than the current terminal atom, replace it
                elif neighbour.atomic_number > terminal_atoms.get(atom).atomic_number:
                    terminal_atoms[atom] = neighbour

    # build out the torsion
    return tuple(
        [
            terminal_atoms[atoms[0]].molecule_atom_index,
            atoms[0].molecule_atom_index,
            atoms[1].molecule_atom_index,
            terminal_atoms[atoms[1]].molecule_atom_index,
        ]
    )


def get_symmetry_classes(molecule: off.Molecule) -> List[int]:
    """Calculate the symmetry classes of each atom in the molecule using the backend toolkits."""

    try:
        from rdkit import Chem

        rd_mol = molecule.to_rdkit()
        symmetry_classes = list(Chem.CanonicalRankAtoms(rd_mol, breakTies=False))

    except (ImportError, ModuleNotFoundError):
        from openeye import oechem

        oe_mol = molecule.to_openeye()
        oechem.OEPerceiveSymmetry(oe_mol)

        symmetry_classes_by_index = {
            a.GetIdx(): a.GetSymmetryClass() for a in oe_mol.GetAtoms()
        }
        symmetry_classes = [
            symmetry_classes_by_index[i] for i in range(molecule.n_atoms)
        ]

    return symmetry_classes


def get_symmetry_group(
    atom_group: Tuple[int, ...], symmetry_classes: List[int]
) -> Tuple[int, ...]:
    """
    For the list of atom groups calculate their symmetry class for the given molecule.
    """
    return tuple([symmetry_classes[atom] for atom in atom_group])
