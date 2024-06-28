from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from qcportal import PortalClient

from openff.qcsubmit.results import OptimizationResultCollection
from openff.qcsubmit.utils import portal_client_manager


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
    """Retrieve a full dataset from QCArchive to populate the cache, then
    access the cache offline in `to_records` by passing the same client args to
    the `portal_client_manager`
    """
    with TemporaryDirectory() as d:
        client = PortalClient("https://api.qcarchive.molssi.org:443", cache_dir=d)
        assert (Path(d) / "api.qcarchive.molssi.org_443").exists()
        opt = OptimizationResultCollection.from_server(
            client,
            "OpenFF Torsion Multiplicity Optimization Training Coverage Supplement v1.0",
        )
        # this fails without the portal_client_manager, as desired. reusing the
        # same client in the lambda prevents the network access that occurs
        # when creating the client
        with portal_client_manager(lambda _: client), no_internet():
            opt.to_records()
        with no_internet(), pytest.raises(Exception, match="socket accessed"):
            opt.to_records()
