"""Small structured logger with stable scientific warning codes."""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import TextIO


@dataclass
class RunLogger:
    """Emit concise console messages and retain warnings for the manifest."""

    verbose: bool = False
    stdout: TextIO = sys.stdout
    stderr: TextIO = sys.stderr
    warnings: list[str] = field(default_factory=list)

    def info(self, message: str) -> None:
        """Emit a normal user-facing status line."""
        print(message, file=self.stdout)

    def detail(self, message: str) -> None:
        """Emit a diagnostic line only in verbose mode."""
        if self.verbose:
            print(message, file=self.stdout)

    def warning(self, code: str, message: str) -> None:
        """Emit and retain a warning with a stable code."""
        formatted = f"[{code}] {message}"
        self.warnings.append(formatted)
        print(formatted, file=self.stderr)
