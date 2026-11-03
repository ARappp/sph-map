"""Allow ``python -m sph`` to invoke the supported command line."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
