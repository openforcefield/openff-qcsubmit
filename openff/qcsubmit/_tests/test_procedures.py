import pytest

from openff.qcsubmit._pydantic import ValidationError
from openff.qcsubmit.procedures import GeometricProcedure


@pytest.mark.parametrize(
    "convergence_set",
    [
        pytest.param(
            "GAU",
            id="Default convergence_set value",
        ),
        pytest.param(
            "GAU_VERYTIGHT",
            id="Very tight convergence_set value",
        ),
        pytest.param(
            "TURBOMOLE",
            id="Turbomole convergence_set value",
        ),
    ],
)
def test_convergence_set(convergence_set):
    """
    Test that the convergence_set is set correctly.
    """
    procedure = GeometricProcedure(convergence_set=convergence_set)
    assert procedure.convergence_set == convergence_set
    assert procedure.get_optimzation_keywords()["convergence_set"] == convergence_set
    assert "converge" not in procedure.get_optimzation_keywords()


@pytest.mark.parametrize(
    "conv_keywords",
    [
        pytest.param(
            (
                "CUSTOM",
                [
                    "energy",
                    "1e-8",
                ],
            ),
            id="Just energy",
        ),
        pytest.param(
            (
                "CUSTOM",
                [
                    "grms",
                    "3e-4",
                    "energy",
                    "1e-6",
                    "drms",
                    "1.2e-3",
                    "gmax",
                    "4.5e-4",
                    "dmax",
                    "1.8e-3",
                ],
            ),
            id="All specified, out of order",
        ),
        pytest.param(
            (
                "CUSTOM",
                ["grms", "3e-4", "energy", "1e-6"],
            ),
            id="Two specified",
        ),
    ],
)
def test_custom_converge(conv_keywords):
    """
    Test that the custom convergence criteria is set correctly.
    """
    convergence_set, converge = conv_keywords
    procedure = GeometricProcedure(convergence_set=convergence_set, converge=converge)
    assert procedure.converge == converge
    assert procedure.get_optimzation_keywords()["converge"] == converge
    assert "convergence_set" not in procedure.get_optimzation_keywords()


@pytest.mark.parametrize(
    "conv_keywords",
    [
        pytest.param(
            ("CUSTOM", ["energy", "1e-8", "maxiter"], 3),
            id="Custom + maxiter",
        ),
        pytest.param(
            ("GAU_VERYTIGHT", ["maxiter"], 3),
            id="Regular + maxiter",
        ),
    ],
)
def test_converge_maxiter(conv_keywords):
    """
    Test that --converge maxiter works with both custom convergence options and convergence_set.
    """
    convergence_set, converge, maxiter = conv_keywords
    procedure = GeometricProcedure(
        convergence_set=convergence_set, converge=converge, maxiter=maxiter
    )
    assert procedure.converge == converge
    assert procedure.get_optimzation_keywords()["converge"] == converge
    assert procedure.convergence_set == convergence_set
    if convergence_set != "CUSTOM":
        assert (
            procedure.get_optimzation_keywords()["convergence_set"] == convergence_set
        )
    else:
        assert "convergence_set" not in procedure.get_optimzation_keywords()
    assert procedure.maxiter == maxiter
    assert procedure.get_optimzation_keywords()["maxiter"] == maxiter


@pytest.mark.parametrize(
    "converge",
    [
        pytest.param(
            "GAU",
        ),
        pytest.param(
            ["energy", 1e-8],
        ),
        pytest.param(["hello", "1e-8"]),
        pytest.param(["energy", "grms", "maxiter"]),
        pytest.param(["energy", "1e-8", "maxiter", "5"]),
        pytest.param(["energy", "hello"]),
        pytest.param(["GAU", "maxiter"]),
    ],
)
def test_converge_validation_errors(converge):
    """
    Make sure that invalid values of converge raise a ValidationError
    """
    with pytest.raises(ValidationError):
        procedure = GeometricProcedure(converge=converge)


@pytest.mark.parametrize(
    "convergence_set",
    [
        pytest.param(
            "HELLO",
        ),
        pytest.param("GAU maxiter"),
    ],
)
def test_convergence_set_validation_errors(convergence_set):
    """
    Make sure that invalid values of convergence_set raise a ValidationError
    """
    with pytest.raises(ValidationError):
        procedure = GeometricProcedure(convergence_set=convergence_set)


def test_convergence_set_attribute_error():
    """
    Make sure that an AttributeError is raised if the user tries to set a converge list as convergence_set
    """
    with pytest.raises(AttributeError):
        procedure = GeometricProcedure(convergence_set=["energy", "1e-8"])


@pytest.mark.parametrize(
    "conv_keywords",
    [
        pytest.param(
            (
                "CUSTOM",
                ["maxiter"],
            ),
            id="Custom but just maxiter",
        ),
        pytest.param(
            ("CUSTOM", []),
            id="Custom with empty list",
        ),
        pytest.param(
            (
                "GAU",
                ["energy", "1e-8"],
            ),
            id="Regular convergence_set but custom converge also provided",
        ),
    ],
)
def test_both_validation_errors(conv_keywords):
    """
    Make sure that if incompatible values of converge and convergence_set are set, a ValidationError is raised
    """
    convergence_set, converge = conv_keywords
    with pytest.raises(ValidationError):
        procedure = GeometricProcedure(
            convergence_set=convergence_set, converge=converge
        )
