"""Transactional output helpers for SPH MTZ and metadata files."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from .errors import OutputError, SPHUserError


def validate_output_paths(
    output_paths: Iterable[str | Path],
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    """Validate output parents and enforce the explicit overwrite policy."""
    paths = tuple(Path(path).expanduser() for path in output_paths)
    duplicates = {path for path in paths if paths.count(path) > 1}
    if duplicates:
        raise OutputError(
            "The same path was selected for multiple outputs: "
            + ", ".join(str(path) for path in sorted(duplicates))
            + ". Choose distinct output paths."
        )
    for path in paths:
        parent = path.parent if str(path.parent) else Path(".")
        if not parent.exists():
            raise OutputError(
                f"Output directory does not exist: {parent}. Create it first or "
                "choose an existing directory."
            )
        if not parent.is_dir():
            raise OutputError(
                f"Output parent is not a directory: {parent}. Choose a valid output path."
            )
        if not os.access(parent, os.W_OK):
            raise OutputError(
                f"Output directory is not writable: {parent}. Change permissions or "
                "choose another directory."
            )
        if path.exists() and not overwrite:
            raise OutputError(
                f"Output already exists: {path}. Choose a new path or pass --overwrite "
                "to replace it explicitly."
            )
        if path.exists() and not path.is_file():
            raise OutputError(
                f"Output target exists but is not a regular file: {path}."
            )
    return paths


def _commit_temporary_file(temp_path: Path, target: Path, overwrite: bool) -> None:
    try:
        if overwrite:
            os.replace(temp_path, target)
        else:
            os.link(temp_path, target)
            temp_path.unlink()
    except FileExistsError as error:
        raise OutputError(
            f"Output appeared during calculation and was not overwritten: {target}. "
            "Choose another path or rerun with --overwrite."
        ) from error
    except OSError as error:
        raise OutputError(
            f"Could not commit output {target}: {error}. The temporary output was "
            "not installed as the requested file."
        ) from error


def commit_temporary_output(
    temporary_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Install an already validated temporary output at its final path."""
    temporary = Path(temporary_path)
    target = Path(output_path).expanduser()
    _commit_temporary_file(temporary, target, overwrite)
    return target


def transactional_write_mtz(dataset, output_path: str | Path, *, overwrite: bool) -> Path:
    """Write an MTZ beside its target and atomically install it after success."""
    target = Path(output_path).expanduser()
    validate_output_paths([target], overwrite=overwrite)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp.mtz", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        dataset.write_mtz(str(temporary))
        _commit_temporary_file(temporary, target, overwrite)
    except OutputError:
        raise
    except Exception as error:
        raise OutputError(
            f"Failed while writing MTZ {target}: {error}. No incomplete new MTZ "
            "was installed."
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def transactional_write_json(
    value: object,
    output_path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Serialize finite JSON to a temporary file and atomically install it."""
    target = Path(output_path).expanduser()
    validate_output_paths([target], overwrite=overwrite)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp.json", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _commit_temporary_file(temporary, target, overwrite)
    except OutputError:
        raise
    except Exception as error:
        raise OutputError(
            f"Failed while writing JSON metadata {target}: {error}. No incomplete "
            "new metadata file was installed."
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def transactional_write_text(
    text: str,
    output_path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Write UTF-8 text transactionally using the same overwrite policy."""
    target = Path(output_path).expanduser()
    validate_output_paths([target], overwrite=overwrite)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp.txt", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _commit_temporary_file(temporary, target, overwrite)
    except OutputError:
        raise
    except Exception as error:
        raise OutputError(
            f"Failed while writing text output {target}: {error}. No incomplete "
            "new file was installed."
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
