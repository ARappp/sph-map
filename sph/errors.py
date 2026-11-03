"""Actionable exception hierarchy for SPH user-facing failures."""


class SPHError(Exception):
    """Base class for errors raised by the SPH package."""


class SPHUserError(SPHError):
    """A recoverable, actionable error caused by input or configuration."""


class ConfigurationError(SPHUserError):
    """Raised when a requested configuration is invalid or unsupported."""


class ReflectionInputError(SPHUserError):
    """Raised when a reflection file cannot be used scientifically."""


class SymmetryInputError(ReflectionInputError):
    """Raised when reflection symmetry content violates the selected policy."""


class OutputError(SPHUserError):
    """Raised when a requested output cannot be written safely."""
