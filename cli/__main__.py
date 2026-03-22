"""Make bench package executable with python -m bench."""

import sys

from cli.main import main

if __name__ == "__main__":
    sys.exit(main())
