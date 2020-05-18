"""
Unit and regression test for the qcsubmit package.
"""

import sys

import pytest

# Import package, test suite, and other packages as needed
import qcsubmit


def test_qcsubmit_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qcsubmit" in sys.modules
