from .. import constants as c
from .reputation_matrix_factorization import ReputationMFModel, train_model

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def get_low_diligence_intercepts(
  filteredRatings: pd.DataFrame,
  noteParams: pd.DataFrame,
  raterParams: pd.DataFrame,
  device=torch.device("cpu"),
) -> pd.DataFrame:
  # Generate input tensors
  noteIdMap = dict(zip(noteParams[c.noteIdKey], np.arange(len(noteParams), dtype=np.int64)))
  raterIdMap = dict(
    zip(raterParams[c.raterParticipantIdKey], np.arange(len(raterParams), dtype=np.int64))
  )
  noteTensor = torch.tensor(
    [noteIdMap[noteId] for noteId in filteredRatings[c.noteIdKey]], device=device
  )
  raterTensor = torch.tensor(
    [raterIdMap[raterId] for raterId in filteredRatings[c.raterParticipantIdKey]], device=device
  )
  targetTensor = torch.tensor(
    (
      filteredRatings[c.notHelpfulIncorrectTagKey]
      + filteredRatings[c.notHelpfulIrrelevantSourcesTagKey]
      + filteredRatings[c.notHelpfulSourcesMissingOrUnreliableTagKey]
    ).clip(0, 1),
    device=device,
    dtype=torch.float32,
  )
  # Compute low diligence intercepts
  model = ReputationMFModel(
    noteParams,
    raterParams,
    l2Lambda=0.03,
    l2NoteBiasMultiplier=1,
    l2RaterBiasMultiplier=10,
    l2RaterReputationMultiplier=50,
    l2GlobalBiasMultiplier=0,
    stableInit=True,
  )
  loss0, loss1, loss2 = train_model(
    model=model,
    dataset=(noteTensor, raterTensor, targetTensor),
    loss_fn=nn.BCEWithLogitsLoss(reduction="none"),
    posWeight=100,
    l2LambdaThirdRoundMultiplier=1,
    posWeightThirdRoundMultiplier=5,
    numEpochs=300,
    logRate=30,
    alpha=0.1,
    raterWeightPower=0.5,
    initialNoteWeightPower=0,
    finalNoteWeightPower=-0.5,
    freezeRaterBias=False,
    resetReputation=False,
  )
  print(f"Low diligence training loss: {loss0:.4f}, {loss1:.4f}, {loss2:.4f}")
  # Compose and return DataFrame
  return pd.DataFrame(
    {
      c.noteIdKey: noteParams[c.noteIdKey],
      c.lowDiligenceInterceptKey: model.noteBias.weight.cpu().flatten().detach().numpy(),
    }
  )
