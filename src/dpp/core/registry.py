from typing import Dict
from dpp.core.models import PricingModel

_REGISTRY: Dict[str, PricingModel] = {}

def register_model(name: str, model: PricingModel) -> None:
    _REGISTRY[name] = model

def get_model(name: str) -> PricingModel:
    if name not in _REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]

def list_models() -> list[str]:
    return list(_REGISTRY.keys())
