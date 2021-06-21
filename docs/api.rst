API
===

Below is an outline of the API for ``openff-qcsubmit`` See the examples for details on how to use these objects.

.. warning:: The ``openff-qcsubmit`` package is still pre-alpha so the API is still in flux.

Datasets
--------

.. currentmodule:: openff.qcsubmit.datasets
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BasicDataset
    DatasetEntry
    OptimizationDataset
    OptimizationEntry
    TorsiondriveDataset
    TorsionDriveEntry
    ComponentResult

Factories
"""""""""

.. currentmodule:: openff.qcsubmit.factories
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseDatasetFactory
    BasicDatasetFactory
    OptimizationDatasetFactory
    TorsiondriveDatasetFactory

Procedures
""""""""""

.. currentmodule:: openff.qcsubmit.procedures
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    GeometricProcedure

Constraints
"""""""""""

.. currentmodule:: openff.qcsubmit.constraints
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Constraint
    DistanceConstraint
    DistanceConstraintSet
    AngleConstraint
    AngleConstraintSet
    DihedralConstraint
    DihedralConstraintSet
    PositionConstraint
    PositionConstraintSet

Results
-------

.. currentmodule:: openff.qcsubmit.results
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BasicResult
    BasicResultCollection
    OptimizationResult
    OptimizationResultCollection
    TorsionDriveResult
    TorsionDriveResultCollection

Workflow Components
-------------------

.. currentmodule:: openff.qcsubmit.workflow_components
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CustomWorkflowComponent
    BasicSettings
    ToolkitValidator

Conformer Generation
""""""""""""""""""""

.. currentmodule:: openff.qcsubmit.workflow_components
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    StandardConformerGenerator

Filters
"""""""

.. currentmodule:: openff.qcsubmit.workflow_components
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CoverageFilter
    ElementFilter
    MolecularWeightFilter
    RMSDCutoffConformerFilter
    RotorFilter
    SmartsFilter

Fragmentation
"""""""""""""

.. currentmodule:: openff.qcsubmit.workflow_components
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PfizerFragmenter
    WBOFragmenter

State Enumeration
"""""""""""""""""

.. currentmodule:: openff.qcsubmit.workflow_components
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    EnumerateProtomers
    EnumerateStereoisomers
    EnumerateTautomers

Common Structures
-----------------

.. currentmodule:: openff.qcsubmit.common_structures
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    DatasetConfig
    ResultsConfig
    ComponentProperties
    TDSettings
    PCMSettings
    QCSpec
    QCSpecificationHandler
    SingleTorsion
    DoubleTorsion
    ImproperTorsion
    TorsionIndexer
    IndexCleaner
    ClientHandler
    Metadata
    MoleculeAttributes
    SCFProperties
    CommonBase

Exceptions
----------

.. currentmodule:: openff.qcsubmit.exceptions
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    QCSubmitException
    UnsupportedFiletypeError
    InvalidWorkflowComponentError
    MissingWorkflowComponentError
    ComponentRegisterError
    ComponentRequirementError
    InvalidClientError
    DriverError
    DatasetInputError
    MissingBasisCoverageError
    DihedralConnectionError
    LinearTorsionError
    MolecularComplexError
    ConstraintError
    DatasetCombinationError
    QCSpecificationError
    AngleConnectionError
    BondConnectionError
    AtomConnectionError
    PCMSettingError
    InvalidDatasetError
    DatasetRegisterError
    RecordTypeError