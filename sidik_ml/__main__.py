"""Module entry point for running SIDIK-ML directly via `python -m sidik_ml`."""

import sys

from sidik_ml.cli import main

if __name__ == "__main__":
    sys.exit(main())
