import logging

import pytest
from pydantic import ValidationError

from openff.qcsubmit.results.filters import (
    ResultFilter,
    ResultRecordFilter,
    SMARTSFilter,
    SMILESFilter,
)


def test_apply_filter(basic_result_collection, caplog):
    class DummyFilter(ResultFilter):
        def _apply(self, result_collection):

            result_collection.entries = {
                "http://localhost:442": result_collection.entries[
                    "http://localhost:442"
                ]
            }

            return result_collection

    with caplog.at_level(logging.INFO):
        filtered_collection = DummyFilter().apply(basic_result_collection)

    assert filtered_collection.n_results == 4
    assert "4 results were removed" in caplog.text

    assert "applied-filters" in filtered_collection.provenance
    assert "DummyFilter-0" in filtered_collection.provenance["applied-filters"]


def test_apply_record_filter(basic_result_collection):
    class DummyFilter(ResultRecordFilter):
        def _filter_function(self, result, record, molecule) -> bool:
            return record.client.address == "http://localhost:442"

    filtered_collection = DummyFilter().apply(basic_result_collection)

    assert filtered_collection.n_results == 4

    assert "http://localhost:442" in filtered_collection.entries
    assert "http://localhost:443" not in filtered_collection.entries


def test_smiles_filter_mutual_inputs():

    with pytest.raises(ValidationError, match="exactly one of `smiles_to_include`"):
        SMILESFilter(smiles_to_include=["C"], smiles_to_exclude=["CC"])


def test_smarts_filter_mutual_inputs():

    with pytest.raises(ValidationError, match="exactly one of `smarts_to_include`"):
        SMARTSFilter(smarts_to_include=["C"], smarts_to_exclude=["CC"])


@pytest.mark.parametrize(
    "result_filter, expected_ids",
    [
        (SMILESFilter(smiles_to_include=["CCO"]), {"http://localhost:442": {"2", "3"}}),
        (
            SMILESFilter(smiles_to_exclude=["CCO"]),
            {
                "http://localhost:442": {"1", "4"},
                "http://localhost:443": {"1", "2", "3", "4"},
            },
        ),
        (
            SMARTSFilter(smarts_to_include=["[#6]-[#8H1]"]),
            {"http://localhost:442": {"1", "2", "3", "4"}},
        ),
        (
            SMARTSFilter(smarts_to_exclude=["[#6]-[#8]"]),
            {"http://localhost:443": {"1", "2", "3", "4"}},
        ),
    ],
)
def test_molecule_filter_apply(result_filter, expected_ids, basic_result_collection):

    filtered_collection = result_filter.apply(basic_result_collection)

    assert {*expected_ids} == {*filtered_collection.entries}

    for address, entry_ids in expected_ids.items():

        assert entry_ids == {
            entry.record_id for entry in filtered_collection.entries[address]
        }
