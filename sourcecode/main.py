#!/usr/bin/env python3
"""Invoke Community Notes scoring and user contribution algorithms.

Example Usage:
  python main.py \
    --enrollment data/userEnrollment-00000.tsv \
    --notes data/notes-00000.tsv \
    --ratings data/ratings-00000.tsv \
    --status data/noteStatusHistory-00000.tsv \
    --outdir data
"""

from scoring.runner import main


if __name__ == "__main__":
  main()
