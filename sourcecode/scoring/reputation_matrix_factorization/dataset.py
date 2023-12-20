from dataclasses import dataclass
from typing import Optional

from .. import constants as c

import numpy as np
import pandas as pd
import torch


@dataclass
class MatrixFactorizationDataset:
  # Tensors specifying the note, rater and target for each rating
  noteTensor: torch.Tensor
  raterTensor: torch.Tensor
  targetTensor: torch.Tensor
  # Ordered notes and raters associated with each index
  notes: pd.DataFrame
  raters: pd.DataFrame


def build_dataset(
  ratings: pd.DataFrame,
  targets: np.ndarray,
  notes: Optional[pd.DataFrame] = None,
  raters: Optional[pd.DataFrame] = None,
  device: torch.device = torch.device("cpu"),
) -> MatrixFactorizationDataset:
  """Compose and return a MatrixFactorizationDataset given ratings and targets.

  Args:
    ratings: DF specifying notes and raters
    targets: numpy array specifying target values
    notes: Optionally, a DF ordering notes for assignment to indices
    raters: Optionally, a DF ordering raters for assignment to indices
    device: torch device where tensors should be stored (e.g. cuda, mps, cpu)
  """
  # Identify mappings from note and rater IDs to indices
  if notes is None:
    notes = ratings[[c.noteIdKey]].drop_duplicates().sort_values()
  noteIdMap = dict(zip(notes[c.noteIdKey], np.arange(len(notes), dtype=np.int64)))
  if raters is None:
    raters = ratings[[c.raterParticipantIdKey]].drop_duplicates().sort_values()
  raterIdMap = dict(zip(raters[c.raterParticipantIdKey], np.arange(len(raters), dtype=np.int64)))
  # Generate tensors
  noteTensor = torch.tensor([noteIdMap[noteId] for noteId in ratings[c.noteIdKey]], device=device)
  raterTensor = torch.tensor(
    [raterIdMap[raterId] for raterId in ratings[c.raterParticipantIdKey]], device=device
  )
  targetTensor = torch.tensor(targets, device=device, dtype=torch.float32)
  # Return MatrixFactorizationDataset
  return MatrixFactorizationDataset(
    noteTensor=noteTensor,
    raterTensor=raterTensor,
    targetTensor=targetTensor,
    notes=notes,
    raters=raters,
  )
