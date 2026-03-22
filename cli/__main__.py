"""Make bench package executable with python -m bench."""

import sys

from bench.cli import main

if __name__ == "__main__":
    sys.exit(main())
