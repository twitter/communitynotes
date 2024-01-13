from typing import Optional

from .. import constants as c
from .dataset import build_dataset
from .reputation_matrix_factorization import ReputationModelHyperparameters, train_model

import pandas as pd
import torch


def get_low_diligence_intercepts(
  filteredRatings: pd.DataFrame,
  noteInitState: Optional[pd.DataFrame] = None,
  raterInitState: Optional[pd.DataFrame] = None,
  device=torch.device("cpu"),
) -> pd.DataFrame:
  # Define dataset
  targets = (
    (
      filteredRatings[c.notHelpfulIncorrectTagKey]
      + filteredRatings[c.notHelpfulIrrelevantSourcesTagKey]
      + filteredRatings[c.notHelpfulSourcesMissingOrUnreliableTagKey]
    )
    .clip(0, 1)
    .values
  )
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
    convergence=10**-5,
    stablePeriod=5,
    # Regularization hyperparameters
    l2Lambda=0.03,
    l2NoteBiasMultiplier=1,
    l2RaterBiasMultiplier=10,
    l2GlobalBiasMultiplier=0,
    l2RaterReputationMultiplier=50,
    l2LambdaThirdRoundMultiplier=1,
    l2NoteBiasThirdRoundMultiplier=1,
    # Base / first round loss hyperparameters
    lossFunction="BCEWithLogitsLoss",
    posWeight=100,
    noteNormExpFirstRound=0,
    raterNormExpFirstRound=0,
    # Second round loss hyperparameters
    posWeightSecondRoundMultiplier=1,
    noteNormExpSecondRound=0,
    raterNormExpSecondRound=0,
    # Third round loss hyperparameters
    posWeightThirdRoundMultiplier=5,
    noteNormExpThirdRound=-0.5,
    raterNormExpThirdRound=0,
    reputationExp=0.5,
    alpha=0.1,
  )

  # Train model
  model, loss1, loss2, loss3 = train_model(
    hParams=hParams,
    dataset=dataset,
    noteInitState=noteInitState,
    raterInitState=raterInitState,
    device=device,
  )
  print(f"Low diligence training loss: {loss1:.4f}, {loss2:.4f}, {loss3:.4f}")

  # Compose and return DataFrame
  return pd.DataFrame(
    {
      c.noteIdKey: dataset.notes,
      c.lowDiligenceInterceptKey: model.noteBias.weight.cpu().flatten().detach().numpy(),
    }
  )
