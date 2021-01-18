"""
The base file with functions to register and de register new workflow components.
"""

from typing import Dict, List, Union

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
    SmartsFilter,
)
from openff.qcsubmit.workflow_components.fragmentation import WBOFragmenter
from openff.qcsubmit.workflow_components.state_enumeration import (
    EnumerateProtomers,
    EnumerateStereoisomers,
    EnumerateTautomers,
)

__all__ = [
    "register_component",
    "deregister_component",
    "get_component",
    "list_components",
]

workflow_components: Dict[str, CustomWorkflowComponent] = {}


def register_component(
    component: CustomWorkflowComponent, replace: bool = False
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

    if issubclass(type(component), CustomWorkflowComponent):
        component_name = component.component_name.lower()
        if component_name not in workflow_components or (
            component_name in workflow_components and replace
        ):
            workflow_components[component_name] = component
        else:
            raise ComponentRegisterError(
                f"There is already a component registered with QCSubmit with the name {component.component_name}, to replace this use the `replace=True` flag."
            )
    else:
        raise InvalidWorkflowComponentError(
            f"Component {component} rejected as it is not a sub class of CustomWorkflowComponent"
        )


def get_component(component_name: str) -> CustomWorkflowComponent:
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


def deregister_component(component: Union[CustomWorkflowComponent, str]) -> None:
    """
    Deregister the workflow component from QCSubmit.

    Parameters:
        component: The name or instance of the component which should be removed.

    Raises:
        ComponentRegisterError: If the component to be removed was not registered.
    """

    if issubclass(type(component), CustomWorkflowComponent):
        component_name = component.component_name.lower()

    else:
        component_name = component.lower()

    wc = workflow_components.pop(component_name, None)
    if wc is None:
        raise ComponentRegisterError(
            f"The component {component} could not be removed as it was not registered."
        )


def list_components() -> List[CustomWorkflowComponent]:
    """
    Get a list of all of the currently registered workflow components.

    Returns:
        A list of the workflow components which are currently registered.
    """

    return list(workflow_components.values())


# register the current components
# conformer generation
register_component(StandardConformerGenerator())

# fragmentation
register_component(WBOFragmenter())

# filters
register_component(RotorFilter())
register_component(RMSDCutoffConformerFilter())
register_component(SmartsFilter())
register_component(CoverageFilter())
register_component(MolecularWeightFilter())
register_component(ElementFilter())

# state enumeration
register_component(EnumerateTautomers())
register_component(EnumerateStereoisomers())
register_component(EnumerateProtomers())
