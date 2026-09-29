# SPDX-License-Identifier: Apache-2.0
"""The frozen app's entry script: ``Proteia.exe`` runs the ``proteia`` console
command (``proteia.web.launch:main``) with its arguments: image paths (the
installer's "Open with Proteia" passes one, #54) and ``--self-test`` included."""

import sys

from proteia.web.launch import main

if __name__ == "__main__":
    sys.exit(main())
