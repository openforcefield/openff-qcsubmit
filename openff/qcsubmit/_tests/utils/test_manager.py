from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from qcportal import PortalClient

from openff.qcsubmit.results import OptimizationResultCollection
from openff.qcsubmit.utils.utils import get_data, portal_client_manager


@contextmanager
def no_internet():
    """A context manager that temporarily disables access to the internet by
    overriding `socket.socket` to raise an exception on use.
    """
    import socket

    orig = socket.socket

    def guard(*args, **kwargs):
        raise Exception("socket accessed")

    socket.socket = guard
    try:
        yield
    finally:
        socket.socket = orig


def test_manager():
    ds = OptimizationResultCollection.parse_file(get_data("tiny-opt-dataset.json"))
    with (
        TemporaryDirectory() as d,
        portal_client_manager(lambda addr: PortalClient(addr, cache_dir=d)),
    ):
        ds.to_records()
        assert (Path(d) / "api.qcarchive.molssi.org_443").exists()
