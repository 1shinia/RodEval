"""AIGC model base class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


def resolve_api_url(api_base: str, tool: str = 'txt2img') -> str:
    """Resolve the full API endpoint URL.

    If api_base already looks like a concrete endpoint, use it as-is.
    Otherwise append the default path based on tool type.
    Edgecloud (aigateway) URLs are used as-is regardless of tool.
    """
    base = api_base.rstrip('/')
    # Edgecloud gateway: use base URL as-is (group suffix is the endpoint)
    if 'edgecloudapp.com' in base:
        return base
    keywords = ['/generations', '/variations', '/edits', '/video']
    if any(k in base for k in keywords):
        return base
    if tool == 'txt2video':
        return f'{base}/video/generations'
    return f'{base}/images/generations'


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
