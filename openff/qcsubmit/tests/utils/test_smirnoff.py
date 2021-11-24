import networkx as nx
import numpy
import pytest

from openff.qcsubmit.utils.smirnoff import (
    combine_openff_molecules,
    split_openff_molecule,
)


def test_split_openff_molecules(imatinib_mesylate):
    """
    Make sure we can correctly split up a multi component openff molecule.
    """

    molecules = split_openff_molecule(molecule=imatinib_mesylate)
    assert len(molecules) == 2


def test_combine_openff_molecules(imatinib_mesylate):
    """
    Make sure we can correctly combine openff molecules into to multi component molecule with the atoms in a continues order.
    """
    # split a mixed molecule and recombine in the correct order
    multi_mol = combine_openff_molecules(split_openff_molecule(imatinib_mesylate))
    # check that each fragment is in a continuous ordering
    for fragment in nx.connected_components(multi_mol.to_networkx()):
        assert sum(numpy.diff(sorted(fragment)) == 1) == len(fragment) - 1
