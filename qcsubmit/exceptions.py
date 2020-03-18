import traceback


class QCSubmitException(Exception):
    """
    Base QCSubmit exception, should always use the appropriate subclass of this exception.
    """

    error_type = "base_error"
    header = "QCSubmit Base Error"

    def __init__(self, message: str):

        super().__init__(message)

        self.raw_message = message
        self.traceback = traceback.format_exc()

    @property
    def error_message(self) -> str:
        return f"{self.header}: {self.raw_message}"


class UnsupportedFiletypeError(QCSubmitException):
    """
    The file type requested is not supported.
    """

    error_type = "file_type_error"
    header = "QCSubmit File Error"


class InvalidWorkflowComponentError(QCSubmitException):
    """
    The workflow component is invalid.
    """

    error_type = "invalid_component_error"
    header = "QCSubmit Workflow Component Error"


class MissingWorkflowComponentError(QCSubmitException):
    """
    The requested workflow component could not be found.
    """

    error_type = "missing_component_error"
    header = "QCSubmit Missing Workflow Component Error"


class InvalidClientError(QCSubmitException):
    """
    The requested client address could not be contacted.
    """

    error_type = "invalid_client_error"
    header = "QCSumit Invalid Client Error"


class DriverError(QCSubmitException):
    """
    The requested driver is not valid.
    """

    error_type = "driver_error"
    header = "QCSubmit Driver Error"
