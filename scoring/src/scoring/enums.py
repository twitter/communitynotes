from enum import Enum, auto
from typing import Set


class Scorers(Enum):
  """Exhaustive list of all scorers to simplify setting enabled/disabled scorers."""

  MFCoreScorer = auto()
  MFCoreWithTopicsScorer = auto()
  MFExpansionScorer = auto()
  # Note that the MFGroupScorer value controls whether *all* group scorers are instantiated,
  # not just a single MFGroupScorer instance.
  MFGroupScorer = auto()
  MFExpansionPlusScorer = auto()
  ReputationScorer = auto()
  MFTopicScorer = auto()
  MFMultiGroupScorer = auto()
  GaussianScorer = auto()


class Topics(Enum):
  """Range of the note topic model."""

  Unassigned = 0
  UkraineConflict = 1
  GazaConflict = 2
  MessiRonaldo = 3
  Scams = 4


def scorers_from_csv(csv: str) -> Set[Scorers]:
  """Converts a CSV of enums to an actual set of Enum values.

  Args:
    csv: CSV string of Scorer names.

  Returns:
    Set containing Scorers.

  Raises:
    ValueError if csv contains a token which is not a valid Scorer.
  """
  values = []
  for value in csv.split(","):
    try:
      values.append(Scorers[value])
    except KeyError:
      raise ValueError(f"Unknown value {value}")
  return set(values)
