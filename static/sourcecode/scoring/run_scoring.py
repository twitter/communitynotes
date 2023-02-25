"""
This file defines "run_scoring" which drives the entire Community Notes scoring process,
including invoking all note scoring algorithms and computing contributor statistics.

run_scoring should be intergrated into main files for execution in internal and external
environments.
"""

from typing import Optional

from . import constants as c
from .algorithm import run_algorithm

import pandas as pd


def run_scoring(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  epochs: int = c.epochs,
  seed: Optional[int] = None,
  pseudoraters: Optional[bool] = True,
):
  """Invokes note scoring algorithms, merges results and computes user stats.

  Args:
    ratings (pd.DataFrame): preprocessed ratings
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    userEnrollment (pd.DataFrame): The enrollment state for each contributor
    epochs (int, optional): number of epochs to train matrix factorization for. Defaults to c.epochs.
    mf_seed (int, optional): if not None, base distinct seeds for the first and second MF rounds on this value
    pseudoraters (bool, optional): if True, compute optional pseudorater confidence intervals

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
      helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
      noteStatusHistory pd.DataFrame: one row per note containing when they got their most recent statuses.
      auxilaryNoteInfo: one row per note containing adjusted and ratio tag values
  """
  return run_algorithm(
    ratings, noteStatusHistory, userEnrollment, epochs=epochs, seed=seed, pseudoraters=pseudoraters
  )
