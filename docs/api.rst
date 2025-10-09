API
===

Below is an outline of the API for ``openff-qcsubmit``. See the examples for details on how to use these objects.

Datasets
""""""""

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
    FilterEntry

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
"""""""

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

Results Filters
"""""""""""""""

.. currentmodule:: openff.qcsubmit.results.filters
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ResultFilter
    CMILESResultFilter
    SMILESFilter
    SMARTSFilter
    ChargeFilter
    ElementFilter
    HydrogenBondFilter
    ConnectivityFilter
    RecordStatusFilter
    UnperceivableStereoFilter
    LowestEnergyFilter
    ConformerRMSDFilter
    MinimumConformersFilter


Workflow Components
"""""""""""""""""""

.. currentmodule:: openff.qcsubmit.workflow_components
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CustomWorkflowComponent
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

    ChargeFilter
    CoverageFilter
    ElementFilter
    MolecularWeightFilter
    RMSDCutoffConformerFilter
    RotorFilter
    SmartsFilter
    ScanFilter

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
    ScanEnumerator

Workflow Utilities
""""""""""""""""""

.. currentmodule:: openff.qcsubmit.workflow_components
.. autosummary::
    :nosignatures:
    :toctree: api/generated

    ComponentResult
    SingleTorsion
    DoubleTorsion
    ImproperTorsion
    TorsionIndexer
    ImproperScan
    Scan1D
    Scan2D
    TorsionIndexer

Common Structures
"""""""""""""""""

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
    IndexCleaner
    Metadata
    MoleculeAttributes
    SCFProperties
    CommonBase

Utilities
"""""""""

.. currentmodule:: openff.qcsubmit.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

	_CachedPortalClient
	portal_client_manager

Exceptions
""""""""""

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
