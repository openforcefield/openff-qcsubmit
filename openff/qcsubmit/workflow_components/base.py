"""
The base file with functions to register and de register new workflow components.
"""

from typing import Dict, List, Type, Union

from openff.qcsubmit.exceptions import (
    ComponentRegisterError,
    InvalidWorkflowComponentError,
)
from openff.qcsubmit.workflow_components.base_component import CustomWorkflowComponent
from openff.qcsubmit.workflow_components.conformer_generation import (
    StandardConformerGenerator,
)
from openff.qcsubmit.workflow_components.filters import (
    CoverageFilter,
    ElementFilter,
    MolecularWeightFilter,
    RMSDCutoffConformerFilter,
    RotorFilter,
    ScanFilter,
    SmartsFilter,
)
from openff.qcsubmit.workflow_components.fragmentation import (
    PfizerFragmenter,
    WBOFragmenter,
)
from openff.qcsubmit.workflow_components.state_enumeration import (
    EnumerateProtomers,
    EnumerateStereoisomers,
    EnumerateTautomers,
    ScanEnumerator,
)

__all__ = [
    "register_component",
    "deregister_component",
    "get_component",
    "list_components",
]

# make a components type
Components = Union[
    StandardConformerGenerator,
    RMSDCutoffConformerFilter,
    CoverageFilter,
    ElementFilter,
    MolecularWeightFilter,
    RotorFilter,
    SmartsFilter,
    EnumerateTautomers,
    EnumerateProtomers,
    EnumerateStereoisomers,
    WBOFragmenter,
    PfizerFragmenter,
    ScanFilter,
    ScanEnumerator,
]


workflow_components: Dict[str, Type[CustomWorkflowComponent]] = {}


def register_component(
    component: Type[CustomWorkflowComponent], replace: bool = False
) -> None:
    """
    Register a valid workflow component with qcsubmit.

    Parameters:
        component: The workflow component to be registered with QCSubmit.
        replace: If the new component should replace any other component with the same name.

    Raises:
        ComponentRegisterError: If a component has already been registered under this name.
        InvalidWorkflowComponentError: If the new component is not a sub class of the bass workflow component.
    """

    if issubclass(component, CustomWorkflowComponent):
        component_name = component.__fields__["type"].default.lower()
        if component_name not in workflow_components or (
            component_name in workflow_components and replace
        ):
            workflow_components[component_name] = component
        else:
            raise ComponentRegisterError(
                f"There is already a component registered with QCSubmit with the name {component.__fields__['type'].default}, to replace this use the `replace=True` flag."
            )
    else:
        raise InvalidWorkflowComponentError(
            f"Component {component} rejected as it is not a sub class of CustomWorkflowComponent"
        )


def get_component(component_name: str) -> Type[CustomWorkflowComponent]:
    """
    Get the registered workflow component by component name.

    Parameters:
        component_name: The name the component is registered as.

    Returns:
        The requested workflow component.

    Raises:
        ComponentRegisterError: If not component is registered under this name.
    """

    component = workflow_components.get(component_name.lower(), None)
    if component is None:
        raise ComponentRegisterError(
            f"No component is registered with QCSubmit under the name {component_name}."
        )

    return component


def deregister_component(component: Union[Type[CustomWorkflowComponent], str]) -> None:
    """
    Deregister the workflow component from QCSubmit.

    Parameters:
        component: The name or instance of the component which should be removed.

    Raises:
        ComponentRegisterError: If the component to be removed was not registered.
    """

    if isinstance(component, str):
        component_name = component.lower()

    else:
        component_name = component.__fields__["type"].default.lower()

    wc = workflow_components.pop(component_name, None)
    if wc is None:
        raise ComponentRegisterError(
            f"The component {component} could not be removed as it was not registered."
        )


def list_components() -> List[Type[CustomWorkflowComponent]]:
    """
    Get a list of all of the currently registered workflow components.

    Returns:
        A list of the workflow components which are currently registered.
    """

    return list(workflow_components.values())


# register the current components
# conformer generation
register_component(StandardConformerGenerator)

# fragmentation
register_component(WBOFragmenter)
register_component(PfizerFragmenter)

# filters
register_component(RotorFilter)
register_component(RMSDCutoffConformerFilter)
register_component(SmartsFilter)
register_component(CoverageFilter)
register_component(MolecularWeightFilter)
register_component(ElementFilter)
register_component(ScanFilter)

# state enumeration
register_component(EnumerateTautomers)
register_component(EnumerateStereoisomers)
register_component(EnumerateProtomers)
