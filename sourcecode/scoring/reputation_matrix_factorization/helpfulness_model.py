from typing import Optional

from .. import constants as c
from .dataset import build_dataset
from .reputation_matrix_factorization import ReputationModelHyperparameters, train_model

import pandas as pd
import torch


def get_helpfulness_reputation_results(
  filteredRatings: pd.DataFrame,
  noteInitState: Optional[pd.DataFrame] = None,
  raterInitState: Optional[pd.DataFrame] = None,
  device=torch.device("cpu"),
) -> pd.DataFrame:
  # Define dataset
  targets = filteredRatings[c.helpfulNumKey].values
  dataset = build_dataset(filteredRatings, targets, device=device)
  # Define hyperparameters
  hParams = ReputationModelHyperparameters(
    # Model hyperparameters
    activationFunction="IDENTITY",
    nDim=1,
    # Optimizaiton hyperparameters
    numEpochs=300,
    logRate=30,
    learningRate=0.2,
    convergence=10**-7,
    stablePeriod=10,
    # Regularization hyperparameters
    l2Lambda=0.01,
    l2NoteBiasMultiplier=0.5,
    l2RaterBiasMultiplier=1.0,
    l2GlobalBiasMultiplier=0.0,
    l2RaterReputationMultiplier=1.0,
    l2LambdaThirdRoundMultiplier=0.1,
    l2NoteBiasThirdRoundMultiplier=10.0,
    # Base / first round loss hyperparameters
    lossFunction="MSELoss",
    posWeight=0.2,
    noteNormExpFirstRound=0.0,
    raterNormExpFirstRound=-0.5,
    # Second round loss hyperparameters
    posWeightSecondRoundMultiplier=5,
    noteNormExpSecondRound=0,
    raterNormExpSecondRound=0,
    # Third round loss hyperparameters
    posWeightThirdRoundMultiplier=5,
    noteNormExpThirdRound=-0.5,
    raterNormExpThirdRound=0,
    reputationExp=1.0,
    alpha=0.0,
  )

  # Train model
  model, loss1, loss2, loss3 = train_model(
    hParams=hParams,
    dataset=dataset,
    noteInitState=noteInitState,
    raterInitState=raterInitState,
    device=device,
  )
  print(f"Helpfulness reputation loss: {loss1:.4f}, {loss2:.4f}, {loss3:.4f}")

  # Compose and return DataFrames
  noteStats = pd.DataFrame(
    {
      c.noteIdKey: dataset.notes,
      c.coverageNoteInterceptKey: model.noteBias.weight.cpu().flatten().detach().numpy(),
      c.coverageNoteFactor1Key: model.noteEmbedding.weight.cpu().flatten().detach().numpy(),
    }
  )
  raterStats = pd.DataFrame(
    {
      c.raterParticipantIdKey: dataset.raters,
      c.raterHelpfulnessReputationKey: model.raterReputation.weight.cpu()
      .flatten()
      .detach()
      .numpy(),
    }
  )
  return noteStats, raterStats
