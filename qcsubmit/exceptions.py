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


class CompoenentRequirementError(QCSubmitException):
    """
    The requested workflow componenet could not be added due to missing requirements.
    """

    error_type = "missing_requirements_error"
    header = "QCSubmit Missing Workflow Component Requirements Error"


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


class DatasetInputError(QCSubmitException):
    """
    The information entered into the dataset is not valid or missing so component.
    """

    error_type = "dataset_input_error"
    header = "Dataset Input Error"


class MissingBasisCoverageError(QCSubmitException):
    """
    The basis set selected does not contain element coverage for all atoms in the dataset.
    """

    error_type = "missing_basis_coverage_error"
    header = "Missing Basis Coverage Error"


class DihedralConnectionError(QCSubmitException):
    """
    The tagged dihedral is not connected on this molecule and should not be driven.
    """

    error_type = "dihedral_connection_error"
    header = "Dihedral Connection Error"


class LinearTorsionError(QCSubmitException):
    """
    The tagged dihedral involves a linear bond which should not be driven.
    """

    error_type = "linear_torsion_error"
    header = "Linear Torsion Error"


class MolecularComplexError(QCSubmitException):
    """
    The molecule is a complex of two or more units.
    """

    error_type = "molecular_complex_error"
    header = "Molecular Complex Error"


class ConstraintError(QCSubmitException):

    error_type = "constraint_error"
    header = "Constraint Error"
