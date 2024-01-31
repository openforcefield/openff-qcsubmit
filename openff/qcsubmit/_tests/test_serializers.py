"""
Test the serializer factory
"""

import pytest

from openff.qcsubmit.exceptions import UnsupportedFiletypeError
from openff.qcsubmit.serializers import (
    BZ2Compressor,
    GzipCompressor,
    JsonDeSerializer,
    JsonSerializer,
    LZMACompressor,
    NoneCompressor,
    YamlDeSerializer,
    YamlSerializer,
    deserialize,
    get_compressor,
    get_deserializer,
    get_serializer,
    register_compressor,
    register_deserializer,
    register_serializer,
    serialize,
    unregister_compressor,
    unregister_deserializer,
    unregister_serializer,
)
from openff.qcsubmit.testing import temp_directory
from openff.qcsubmit.utils import get_data


@pytest.mark.parametrize(
    "serializer_type",
    [
        pytest.param(("JSON", JsonSerializer), id="Json"),
        pytest.param(("YAML", YamlSerializer), id="Yaml"),
    ],
)
def test_register_again(serializer_type):
    """
    Test adding another serializer
    """

    with pytest.raises(ValueError):
        register_serializer(*serializer_type)


@pytest.mark.parametrize(
    "compressor_type",
    [
        pytest.param(("XZ", LZMACompressor), id="XZ"),
        pytest.param(("BZ2", BZ2Compressor), id="BZ2"),
        pytest.param(("GZ", GzipCompressor), id="GZ"),
        pytest.param(("", NoneCompressor), id="Normal"),
    ],
)
def test_register_compressor_again(compressor_type):
    """
    Test adding a compressor again.
    """
    with pytest.raises(ValueError):
        register_compressor(*compressor_type)


@pytest.mark.parametrize(
    "deserializer_type",
    [
        pytest.param(("JSON", JsonDeSerializer), id="Json"),
        pytest.param(("YAML", YamlDeSerializer), id="Yaml"),
    ],
)
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


def test_get_compressor_error():
    """
    Test getting a compressor which is not registered.
    """
    with pytest.raises(UnsupportedFiletypeError):
        get_compressor("zip")


def test_get_deserializer_error():
    """
    Test getting a deserialzier which is not registered.
    """
    with pytest.raises(UnsupportedFiletypeError):
        get_deserializer("bson")


def test_deregister_compressor_error():
    """
    Test removing a compressor that is not registered.
    """
    with pytest.raises(KeyError):
        unregister_compressor("zip")


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

    with pytest.raises(FileNotFoundError):
        deserialize("missing_file.json")


@pytest.mark.parametrize(
    "serializer", [pytest.param(".json", id="Json"), pytest.param(".yaml", id="yaml")]
)
def test_serializer_round_trips(serializer):
    """
    Test serializing data to and from file with no compression.
    """
    # get data in a dict format
    data = deserialize(get_data("settings_with_workflow.json"))
    file_name = "settings_with_workflow" + serializer
    # now export to file and back
    with temp_directory():
        serialize(serializable=data, file_name=file_name, compression=None)
        deserialized_data = deserialize(file_name=file_name)
        assert data == deserialized_data


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param("xz", id="lzma"),
        pytest.param("gz", id="gzip"),
        pytest.param("", id="No compression"),
        pytest.param("bz2", id="bz2"),
    ],
)
@pytest.mark.parametrize(
    "serialization", [pytest.param("json", id="Json"), pytest.param("yaml", id="yaml")]
)
def test_compression_serialization_round_trip_file_name(serialization, compression):
    """
    Test all of the different serialization and compression combinations.
    Here the compression is in the file name.
    """
    # get data in a dict format
    data = deserialize(get_data("settings_with_workflow.json"))
    file_name = "".join(
        ["settings_with_workflow", ".", serialization, ".", compression]
    )
    # now export the file and read back
    with temp_directory():
        serialize(serializable=data, file_name=file_name, compression=None)
        deserialized_data = deserialize(file_name=file_name)
        assert data == deserialized_data


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param("xz", id="lzma"),
        pytest.param("gz", id="gzip"),
        pytest.param("", id="No compression"),
        pytest.param("bz2", id="bz2"),
    ],
)
@pytest.mark.parametrize(
    "serialization", [pytest.param("json", id="Json"), pytest.param("yaml", id="yaml")]
)
def test_compression_serialization_round_trip(serialization, compression):
    """
    Test all of the different serialization and compression combinations.
    Here the compression is given separately
    """
    # get data in a dict format
    data = deserialize(get_data("settings_with_workflow.json"))
    file_name = "".join(["settings_with_workflow", ".", serialization])
    # now export the file and read back
    with temp_directory():
        serialize(serializable=data, file_name=file_name, compression=compression)
        if compression != "":
            file_name += "." + compression
        deserialized_data = deserialize(file_name=file_name)
        assert data == deserialized_data
