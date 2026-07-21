"""Progress tracking utilities for AIGC evaluation."""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track and persist evaluation progress."""

    def __init__(self, progress_file: Path, total: int):
        self.progress_file = progress_file
        self.total = total
        self.current = 0
        self.status = 'running'

    def update(self, current: int, status: Optional[str] = None) -> None:
        """Update progress.

        Args:
            current: Current progress count
            status: Optional status string
        """
        self.current = current
        if status:
            self.status = status

        progress = {
            'current': self.current,
            'total': self.total,
            'percent': (self.current / self.total * 100) if self.total > 0 else 0,
            'status': self.status,
        }

        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)

    def complete(self) -> None:
        """Mark progress as complete."""
        self.update(self.total, status='completed')
        logger.info('Evaluation completed')

    def fail(self, error: str) -> None:
        """Mark progress as failed.

        Args:
            error: Error message
        """
        self.status = 'failed'
        progress = {
            'current': self.current,
            'total': self.total,
            'percent': (self.current / self.total * 100) if self.total > 0 else 0,
            'status': self.status,
            'error': error,
        }

        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)

        logger.error(f'Evaluation failed: {error}')
