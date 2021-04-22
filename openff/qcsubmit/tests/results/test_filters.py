import logging

from openff.qcsubmit.results.filters import ResultFilter


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
