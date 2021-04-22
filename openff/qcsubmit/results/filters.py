import abc
import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:

    from openff.qcsubmit.results.results import _BaseResultCollection

    T = TypeVar("T", bound=_BaseResultCollection)

logger = logging.getLogger(__name__)


class ResultFilter(BaseModel, abc.ABC):
    """The base class for a filter which will retain selection of QC records based on
    a specific criterion.
    """

    @abc.abstractmethod
    def _apply(self, result_collection: "T") -> "T":
        """The internal implementation of thr ``apply`` method which should apply this
        filter to a results collection and return a new collection containing only the
        retained entries.

        Notes:
            The ``result_collection`` passed to this function will be a copy and
            so can be modified in place if needed.

        Args:
            result_collection: The collection to apply the filter to.

        Returns:
            The collection containing only the retained entries.
        """
        raise NotImplementedError()

    def apply(self, result_collection: "T") -> "T":
        """Apply this filter to a results collection, returning a new collection
        containing only the retained entries.

        Args:
            result_collection: The collection to apply the filter to.

        Returns:
            The collection containing only the retained entries.
        """

        filtered_collection = self._apply(result_collection.copy(deep=True))

        logger.info(
            f"{abs(filtered_collection.n_results - result_collection.n_results)} "
            f"results were removed after applying a {self.__class__.__name__} filter."
        )

        return filtered_collection
