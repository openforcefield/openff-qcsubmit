"""
Test the serializer factory
"""
import pytest

from qcsubmit.exceptions import UnsupportedFiletypeError
from qcsubmit.serializers import (
    JsonDeSerializer,
    JsonSerializer,
    YamlDeSerializer,
    YamlSerializer,
    deserialize,
    get_deserializer,
    get_serializer,
    register_deserializer,
    register_serializer,
    unregister_deserializer,
    unregister_serializer,
)


@pytest.mark.parametrize("serializer_type", [
    pytest.param(("JSON", JsonSerializer), id="Json"),
    pytest.param(("YAML", YamlSerializer), id="Yaml")
])
def test_register_again(serializer_type):
    """
    Test adding another serializer
    """

    with pytest.raises(ValueError):
        register_serializer(*serializer_type)


@pytest.mark.parametrize("deserializer_type", [
    pytest.param(("JSON", JsonDeSerializer), id="Json"),
    pytest.param(("YAML", YamlDeSerializer), id="Yaml")
])
def test_register_again_deserializer(deserializer_type):
    """
    Test registering a deserializer again.
    """

    with pytest.raises(ValueError):
        register_deserializer(*deserializer_type)


def test_get_serializer_error():
    """
    Test getting a serializer which is not registered.
    """
    with pytest.raises(UnsupportedFiletypeError):
        get_serializer("bson")


def test_get_deserializer_error():
    """
    Test getting a deserialzier which is not registered.
    """
    with pytest.raises(UnsupportedFiletypeError):
        get_deserializer("bson")


def test_deregister_serializer_error():
    """
    Test removing a serializer that is not registered.
    """
    with pytest.raises(KeyError):
        unregister_serializer("bson")


def test_deregister_deserializer_error():
    """
    Test removing a serializer that is not registered.
    """
    with pytest.raises(KeyError):
        unregister_deserializer("bson")


def test_deserializer_error():
    """
    Test deserialize a missing file.
    """

    with pytest.raises(RuntimeError):
        deserialize("missing_file.json")