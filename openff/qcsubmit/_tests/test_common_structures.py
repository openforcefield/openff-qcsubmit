import builtins

import pytest
from openff.toolkit.topology import Molecule

from openff.qcsubmit._tests import does_not_raise
from openff.qcsubmit.common_structures import Metadata, MoleculeAttributes, QCSpec
from openff.qcsubmit.exceptions import DatasetInputError, QCSpecificationError


def _block_import(*module_prefixes, error=ModuleNotFoundError):
    """
    Return a stand-in for `builtins.__import__` that raises `error` for any
    import whose name starts with one of the given prefixes, and otherwise
    behaves normally.

    The default (`ModuleNotFoundError`) simulates an optional dependency
    (e.g. openmmforcefields) not being installed. Passing
    `error=AssertionError` instead proves that an import is never *attempted*
    at all, rather than merely that a resulting ModuleNotFoundError doesn't
    propagate.
    """
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if any(name.startswith(prefix) for prefix in module_prefixes):
            raise error(f"Import of {name!r} blocked for testing")
        return real_import(name, *args, **kwargs)

    return blocked_import


def test_attributes_from_openff_molecule():
    mol = Molecule.from_smiles("CC")

    attributes = MoleculeAttributes.from_openff_molecule(mol)

    # now make our own cmiles
    test_cmiles = {
        "canonical_smiles": mol.to_smiles(
            isomeric=False, explicit_hydrogens=False, mapped=False
        ),
        "canonical_isomeric_smiles": mol.to_smiles(
            isomeric=True, explicit_hydrogens=False, mapped=False
        ),
        "canonical_explicit_hydrogen_smiles": mol.to_smiles(
            isomeric=False, explicit_hydrogens=True, mapped=False
        ),
        "canonical_isomeric_explicit_hydrogen_smiles": mol.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=False
        ),
        "canonical_isomeric_explicit_hydrogen_mapped_smiles": mol.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=True
        ),
        "molecular_formula": mol.hill_formula,
        "standard_inchi": mol.to_inchi(fixed_hydrogens=False),
        "inchi_key": mol.to_inchikey(fixed_hydrogens=False),
        "fixed_hydrogen_inchi": mol.to_inchi(fixed_hydrogens=True),
        "fixed_hydrogen_inchi_key": mol.to_inchikey(fixed_hydrogens=True),
        "unique_fixed_hydrogen_inchi_keys": {mol.to_inchikey(fixed_hydrogens=True)},
    }
    assert test_cmiles == attributes


def test_attributes_from_openff_with_map():
    """
    Make sure we can provide a valid cmiles for a molecule with an atom_map and ensure the map is not removed.
    """

    mol = Molecule.from_smiles("CC")
    atom_map = {0: 0, 1: 1, 2: 2, 3: 3}
    mol.properties["atom_map"] = atom_map
    cmiles = MoleculeAttributes.from_openff_molecule(molecule=mol)
    assert "atom_map" in mol.properties
    _ = cmiles.to_openff_molecule()


def test_attributes_from_openff_multi_component():
    """
    Make sure the unique inchi keys are updated correctly.
    """
    mol = Molecule.from_smiles(
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5.CS(=O)(=O)O"
    )

    attributes = MoleculeAttributes.from_openff_molecule(mol)
    assert len(attributes.unique_fixed_hydrogen_inchi_keys) == 2


def test_attributes_to_openff_molecule():
    """Round trip a molecule to and from its attributes."""

    mol: Molecule = Molecule.from_smiles("CC")

    attributes = MoleculeAttributes.from_openff_molecule(molecule=mol)

    mol2 = attributes.to_openff_molecule()

    isomorphic, atom_map = Molecule.are_isomorphic(mol, mol2, return_atom_map=True)
    assert isomorphic is True
    # make sure the molecules are in the same order
    assert atom_map == dict((i, i) for i in range(mol.n_atoms))


@pytest.mark.parametrize(
    "metadata, expected_raises",
    [
        (
            Metadata(
                collection_type="torsiondrive",
                dataset_name="ABC",
                short_description="abcdefgh",
                long_description_url="https://github.com/openforcefield",
                long_description="abcdefgh",
                elements={"C", "H"},
            ),
            does_not_raise(),
        ),
        (
            Metadata(
                collection_type="torsiondrive",
                dataset_name="ABC",
                short_description="abcdefgh",
                long_description="abcdefgh",
                elements={"C", "H"},
            ),
            does_not_raise(),
        ),
        (
            Metadata(),
            pytest.raises(
                DatasetInputError,
                match="The metadata has the following incomplete fields",
            ),
        ),
    ],
)
def test_validate_metadata(metadata, expected_raises):
    with expected_raises:
        metadata.validate_metadata(raise_errors=True)


def test_scf_prop_validation():
    """
    Make sure unsupported scf properties are not allowed into a QCSpec.
    """

    with pytest.raises(QCSpecificationError):
        QCSpec(scf_properties=["ddec_charges"])


@pytest.mark.parametrize(
    "method",
    [
        "openff_unconstrained-1.0.0.offxml",
        "openff-1.0.0.offxml",
        "openff-1.0.0",
    ],
)
def test_openmm_method_preserves_offxml_extension(method):
    """
    Regression test: QCSpec used to unconditionally strip a trailing ".offxml"
    off of `method` before storing it. This silently turned a request for the
    *constrained* force field variant (e.g. "openff-1.0.0.offxml") into the
    bare shorthand "openff-1.0.0", which openmmforcefields >=0.16 treats
    differently than <0.16. The user-provided spelling must survive unchanged. This
    requires openmmforcefields to be installed, since validation (and thus
    any normalization of `method`) is skipped entirely without it.
    """
    pytest.importorskip("openmmforcefields")
    spec = QCSpec(method=method, basis="smirnoff", program="openmm")
    assert spec.method == method


def test_bare_unconstrained_method_rejected():
    """
    Starting in openmmforcefields 0.16.0,
    "openff_unconstrained-X.Y.Z" (no .offxml extension) is not a name that
    openmmforcefields' SMIRNOFFTemplateGenerator recognizes: it isn't in the
    curated shorthand list (bare "openff-X.Y.Z" names, which already resolve
    to the unconstrained variant) and it isn't a valid filename on its own.
    This should be rejected here, rather than passing validation and later
    failing deep inside SystemGenerator with a confusing error. This requires
    openmmforcefields to be installed, since validation is skipped entirely
    without it (see test_openmm_validation_skipped_without_openmmforcefields).
    """
    pytest.importorskip("openmmforcefields")
    with pytest.raises(QCSpecificationError):
        QCSpec(method="openff_unconstrained-1.0.0", basis="smirnoff", program="openmm")


def test_gaff_method_is_accepted():
    pytest.importorskip("openmmforcefields")
    spec = QCSpec(method="gaff-2.2.20", basis="antechamber", program="openmm")
    assert spec.method == "gaff-2.2.20"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            dict(method="openff-2.3.0", basis="not-a-real-basis", program="openmm"),
            "The Basis not-a-real-basis is not supported",
        ),
        (
            dict(method="uff", basis="not-none", program="rdkit"),
            r"choose from \(None,\)",
        ),
    ],
)
def test_qcspec_invalid_combinations_raise(kwargs, match):
    """
    Test of basis/program pemutations that don't require openmmforcefields to be installed for validation.
    Note that erroring on an invalid openmm BASIS can be done without the openmmforcefields package, but
    checking the METHOD for a valid basis will attempt to use openmmforcefields (see tests below)
    """
    with pytest.raises(QCSpecificationError, match=match):
        QCSpec(**kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            dict(method="not-a-real-forcefield", basis="smirnoff", program="openmm"),
            "The method not-a-real-forcefield is not supported",
        ),
        (
            dict(
                method="not-a-real-gaff-version",
                basis="antechamber",
                program="openmm",
            ),
            "The method not-a-real-gaff-version is not supported",
        ),
    ],
)
def test_qcspec_invalid_openmm_method_raises(kwargs, match):
    """
    These cases are only rejected once `method` is checked against the
    force fields openmmforcefields knows about, so they require it to be
    installed (see test_openmm_validation_skipped_without_openmmforcefields
    for what happens to invalid methods when it isn't).
    """
    pytest.importorskip("openmmforcefields")
    with pytest.raises(QCSpecificationError, match=match):
        QCSpec(**kwargs)


def test_openmm_validation_skipped_without_openmmforcefields(monkeypatch):
    """
    openmmforcefields is an optional dependency. If it isn't installed, QCSpec
    construction must still succeed - just with method-name validation
    skipped and a warning raised - rather than a hard failure.
    """
    monkeypatch.setattr(builtins, "__import__", _block_import("openmmforcefields"))

    with pytest.warns(UserWarning, match="openmmforcefields is not installed"):
        spec = QCSpec(method="openff-2.3.0", basis="smirnoff", program="openmm")

    assert spec.method == "openff-2.3.0"


def test_openmm_spec_deserialization_without_openmmforcefields(monkeypatch):
    """
    QCSpec validation runs on every pydantic deserialization of a stored
    dataset (it's a pydantic field type, not just a constructor called at
    submission time), not just at first construction. Loading a
    previously-created OpenMM/SMIRNOFF dataset spec must not require
    openmmforcefields to be installed, or simply inspecting/retrieving an old
    dataset becomes impossible on a lightweight (openff-toolkit-only) install.
    """
    monkeypatch.setattr(builtins, "__import__", _block_import("openmmforcefields"))

    with pytest.warns(UserWarning, match="openmmforcefields is not installed"):
        spec = QCSpec.parse_obj(
            {"method": "openff-2.3.0", "basis": "smirnoff", "program": "openmm"}
        )

    assert spec.method == "openff-2.3.0"


def test_default_qcspec_does_not_require_openmmforcefields(monkeypatch):
    """
    The default (psi4) QCSpec, and any other non-"openmm" program, must never
    even attempt to import openmmforcefields: it's an optional dependency, and
    the overwhelming majority of QCSpecs - including the bare `QCSpec()` used
    when retrieving/exporting datasets - have nothing to do with OpenMM. Using
    `error=AssertionError` here proves the import is never attempted, not
    merely that a resulting ModuleNotFoundError doesn't propagate.
    """
    monkeypatch.setattr(
        builtins, "__import__", _block_import("openmmforcefields", error=AssertionError)
    )

    QCSpec()  # should not raise


def test_unrelated_module_not_found_error_is_not_masked(monkeypatch):
    """
    Regression test: the ModuleNotFoundError guard around the
    openmmforcefields import must not be so wide that it also swallows an
    unrelated ModuleNotFoundError raised later while building the
    allowed-methods list (e.g. from a broken third-party offxml plugin during
    force field discovery). Such a failure must propagate normally, not be
    silently reinterpreted as "openmmforcefields is not installed". Requires
    openmmforcefields to actually be installed, since this test is exercising
    what happens *after* that import succeeds.
    """
    pytest.importorskip("openmmforcefields")
    import openff.toolkit.typing.engines.smirnoff as smirnoff_mod

    def broken_get_available_force_fields():
        raise ModuleNotFoundError("No module named 'some_unrelated_broken_plugin'")

    monkeypatch.setattr(
        smirnoff_mod, "get_available_force_fields", broken_get_available_force_fields
    )

    with pytest.raises(ModuleNotFoundError, match="some_unrelated_broken_plugin"):
        QCSpec(method="openff-2.3.0.offxml", basis="smirnoff", program="openmm")
