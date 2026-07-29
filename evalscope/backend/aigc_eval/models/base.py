"""AIGC model base class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AIGCModelBase(ABC):
    """Base class for AIGC model adapters."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name_or_path', '')
        self.device = config.get('device', 'cuda')
        self.dtype = config.get('dtype', 'float16')

    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        """Generate media from prompts."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources."""
        pass
