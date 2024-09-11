import logging
from typing import Optional, Tuple

from .. import constants as c
from .dataset import build_dataset
from .reputation_matrix_factorization import (
  ReputationModelHyperparameters,
  train_model_final,
  train_model_prescoring,
)

import pandas as pd
import torch


logger = logging.getLogger("birdwatch.helpfulness_model")
logger.setLevel(logging.INFO)


def _setup_dataset_and_hparams(
  filteredRatings: pd.DataFrame,
  device=torch.device("cpu"),
):
  # Define dataset
  targets = filteredRatings[c.helpfulNumKey].values
  dataset = build_dataset(filteredRatings, targets, device=device)
  # Define hyperparameters
  hParams = ReputationModelHyperparameters(
    # Model hyperparameters
    activationFunction="IDENTITY",
    nDim=1,
    # Optimization hyperparameters
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
  return dataset, hParams


def get_helpfulness_reputation_results_final(
  filteredRatings: pd.DataFrame,
  noteInitState: pd.DataFrame,
  raterInitState: pd.DataFrame,
  globalIntercept: c.ReputationGlobalIntercept,
  device=torch.device("cpu"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  dataset, hParams = _setup_dataset_and_hparams(filteredRatings, device)

  # Hack: convert "diligenceRound2" to internal round 2, since the diligence fields are used as placeholders
  #   for the helpfulness-reputation model's internal round 2 score, since this model is not used as a
  #   prod scorer now and doesn't run its own diligence model.
  noteInitState[c.internalNoteInterceptRound2Key] = noteInitState[
    c.lowDiligenceNoteInterceptRound2Key
  ]
  raterInitState[c.internalRaterInterceptRound2Key] = raterInitState[
    c.lowDiligenceRaterInterceptRound2Key
  ]

  # Train model
  model, loss = train_model_final(
    hParams=hParams,
    dataset=dataset,
    noteInitState=noteInitState,
    raterInitState=raterInitState,
    globalInterceptInit=globalIntercept,
    device=device,
  )
  logger.info(f"Helpfulness reputation loss: {loss:.4f}")

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


def get_helpfulness_reputation_results_prescoring(
  filteredRatings: pd.DataFrame,
  noteInitState: Optional[pd.DataFrame] = None,
  raterInitState: Optional[pd.DataFrame] = None,
  device=torch.device("cpu"),
) -> Tuple[pd.DataFrame, pd.DataFrame, c.ReputationGlobalIntercept]:
  dataset, hParams = _setup_dataset_and_hparams(filteredRatings, device)

  # Train model
  (
    model,
    loss1,
    loss2,
    loss3,
    globalIntercept,
    noteIntercept2,
    raterIntercept2,
  ) = train_model_prescoring(
    hParams=hParams,
    dataset=dataset,
    noteInitState=noteInitState,
    raterInitState=raterInitState,
    device=device,
  )
  logger.info(f"Helpfulness reputation loss: {loss1:.4f}, {loss2:.4f}, {loss3:.4f}")

  # Compose and return DataFrames
  noteStats = pd.DataFrame(
    {
      c.noteIdKey: dataset.notes,
      c.internalNoteInterceptKey: model.noteBias.weight.cpu().flatten().detach().numpy(),
      c.internalNoteFactor1Key: model.noteEmbedding.weight.cpu().flatten().detach().numpy(),
      # Hack for now: not actually diligence, but it's the 2nd round intercept from helpfulness.
      # TODO: make a new top-level field for 2nd round reputation model intercepts in top-level prescoring output.
      c.lowDiligenceNoteInterceptRound2Key: noteIntercept2,
    }
  )
  raterStats = pd.DataFrame(
    {
      c.raterParticipantIdKey: dataset.raters,
      c.internalRaterReputationKey: model.raterReputation.weight.cpu().flatten().detach().numpy(),
      c.internalRaterInterceptKey: model.raterBias.weight.cpu().flatten().detach().numpy(),
      c.internalRaterFactor1Key: model.raterEmbedding.weight.cpu().flatten().detach().numpy(),
      # Hack for now: not actually diligence, but it's the 2nd round intercept from helpfulness.
      # TODO: make a new top-level field for 2nd round reputation model intercepts in top-level prescoring output.
      c.lowDiligenceRaterInterceptRound2Key: raterIntercept2,
    }
  )
  return noteStats, raterStats, globalIntercept
