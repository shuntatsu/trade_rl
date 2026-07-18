#!/usr/bin/env python3
"""Compatibility launcher for the explicit fresh-confirmation finalization phase."""

from __future__ import annotations

import sys

from run_full_research_state import main

if __name__ == "__main__":
    raise SystemExit(main(["--phase", "finalize", *sys.argv[1:]]))
