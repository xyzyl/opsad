from .base import DomainAdapter, characterize_interval
from .custom import CustomAdapter
from .energy import EnergyGridAdapter
from .ocean import OceanAdapter
from .plasma import PlasmaAdapter
from .satellite import SatelliteAdapter

ADAPTER_REGISTRY = {
    "plasma": PlasmaAdapter,
    "ocean": OceanAdapter,
    "satellite": SatelliteAdapter,
    "energy": EnergyGridAdapter,
}


def get_adapter(domain: str) -> DomainAdapter:
    """Instantiate the adapter for a domain name, or raise KeyError."""
    return ADAPTER_REGISTRY[domain]()


__all__ = [
    "DomainAdapter",
    "characterize_interval",
    "PlasmaAdapter",
    "OceanAdapter",
    "SatelliteAdapter",
    "EnergyGridAdapter",
    "CustomAdapter",
    "ADAPTER_REGISTRY",
    "get_adapter",
]
