"""Physics engines required by the public TGSR High-Peclet experiment."""

from .base_physics import Physics
from .heat_2d_convection import Heat2DConvection
from .heat_2d_inverse import Heat2DInverse


PHYSICS_REGISTRY = {
    "heat_2d_inverse": Heat2DInverse,
    "heat_2d_convection": Heat2DConvection,
}


def get_physics_engine(config):
    physics_type = config.get("type")
    if physics_type not in PHYSICS_REGISTRY:
        raise ValueError(
            f"Unknown physics type: {physics_type}. "
            f"Available public physics: {list(PHYSICS_REGISTRY.keys())}"
        )
    return PHYSICS_REGISTRY[physics_type](config)


__all__ = [
    "Physics",
    "Heat2DConvection",
    "Heat2DInverse",
    "PHYSICS_REGISTRY",
    "get_physics_engine",
]
