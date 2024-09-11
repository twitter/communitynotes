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


logger = logging.getLogger("birdwatch.diligence_model")
logger.setLevel(logging.INFO)


def _setup_dataset_and_hparams(
  filteredRatings: pd.DataFrame,
  device=torch.device("cpu"),
  ratingsPerNoteLossRatio: Optional[float] = None,
  ratingsPerUserLossRatio: Optional[float] = None,
):
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
    # Optimization hyperparameters
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
    defaultReputation=1.0,
    ratingPerNoteLossRatio=ratingsPerNoteLossRatio,  # 35.0, # approx 29377568 / 795977
    ratingPerUserLossRatio=ratingsPerUserLossRatio,  # 75.0, # approx 29377568 / 265214
  )
  return dataset, hParams


def _prepare_diligence_init_state(noteInitState, raterInitState):
  if noteInitState is not None:
    noteInitState = noteInitState[
      [c.noteIdKey] + [col for col in noteInitState.columns if "lowDiligence" in col]
    ]
    noteInitState.columns = [
      col.replace("lowDiligence", "internal") for col in noteInitState.columns
    ]
  if raterInitState is not None:
    raterInitState = raterInitState[
      [c.raterParticipantIdKey] + [col for col in raterInitState.columns if "lowDiligence" in col]
    ]
    raterInitState.columns = [
      col.replace("lowDiligence", "internal") for col in raterInitState.columns
    ]
  return noteInitState, raterInitState


def fit_low_diligence_model_final(
  filteredRatings: pd.DataFrame,
  noteInitStateDiligence: pd.DataFrame,
  raterInitStateDiligence: pd.DataFrame,
  globalInterceptDiligence: c.ReputationGlobalIntercept,
  ratingsPerNoteLossRatio: Optional[float] = None,
  ratingsPerUserLossRatio: Optional[float] = None,
  device=torch.device("cpu"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Args:
    filteredRatings: DataFrame containing ratings data
    noteInitStateDiligence: DataFrame containing initial state for notes (expects diligence prefixes e.g. lowDiligenceNoteIntercept)
    raterInitStateDiligence: DataFrame containing initial state for raters (expects diligence prefixes, not internal prefixes)
    globalInterceptDiligence: float
    device: torch.device to use for training
  """
  dataset, hParams = _setup_dataset_and_hparams(
    filteredRatings, device, ratingsPerNoteLossRatio, ratingsPerUserLossRatio
  )
  noteInitStateInternal, raterInitStateInternal = _prepare_diligence_init_state(
    noteInitStateDiligence, raterInitStateDiligence
  )

  model, loss_final = train_model_final(
    hParams=hParams,
    dataset=dataset,
    noteInitState=noteInitStateInternal,
    raterInitState=raterInitStateInternal,
    globalInterceptInit=globalInterceptDiligence,
    device=device,
  )
  logger.info(f"Low diligence final loss: {loss_final:.4f}")

  noteStats = pd.DataFrame(
    {
      c.noteIdKey: dataset.notes,
      c.lowDiligenceNoteInterceptKey: model.noteBias.weight.cpu().flatten().detach().numpy(),
      c.lowDiligenceNoteFactor1Key: model.noteEmbedding.weight.cpu().flatten().detach().numpy(),
    }
  )
  raterStats = pd.DataFrame(
    {
      c.raterParticipantIdKey: dataset.raters,
      c.lowDiligenceRaterInterceptKey: model.raterBias.weight.cpu().flatten().detach().numpy(),
      c.lowDiligenceRaterReputationKey: model.raterReputation.weight.cpu()
      .flatten()
      .detach()
      .numpy(),
      c.lowDiligenceRaterFactor1Key: model.raterEmbedding.weight.cpu().flatten().detach().numpy(),
    }
  )
  return noteStats, raterStats


def fit_low_diligence_model_prescoring(
  filteredRatings: pd.DataFrame,
  noteInitStateDiligence: Optional[pd.DataFrame] = None,
  raterInitStateDiligence: Optional[pd.DataFrame] = None,
  device=torch.device("cpu"),
) -> Tuple[pd.DataFrame, pd.DataFrame, c.ReputationGlobalIntercept]:
  dataset, hParams = _setup_dataset_and_hparams(filteredRatings, device)
  noteInitStateInternal, raterInitStateInternal = _prepare_diligence_init_state(
    noteInitStateDiligence, raterInitStateDiligence
  )

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
    noteInitState=noteInitStateInternal,
    raterInitState=raterInitStateInternal,
    device=device,
  )
  logger.info(f"Low diligence training loss: {loss1:.4f}, {loss2:.4f}, {loss3:.4f}")

  noteStats = pd.DataFrame(
    {
      c.noteIdKey: dataset.notes,
      c.lowDiligenceNoteInterceptKey: model.noteBias.weight.cpu().flatten().detach().numpy(),
      c.lowDiligenceNoteFactor1Key: model.noteEmbedding.weight.cpu().flatten().detach().numpy(),
      c.lowDiligenceNoteInterceptRound2Key: noteIntercept2,
    }
  )
  raterStats = pd.DataFrame(
    {
      c.raterParticipantIdKey: dataset.raters,
      c.lowDiligenceRaterInterceptKey: model.raterBias.weight.cpu().flatten().detach().numpy(),
      c.lowDiligenceRaterReputationKey: model.raterReputation.weight.cpu()
      .flatten()
      .detach()
      .numpy(),
      c.lowDiligenceRaterFactor1Key: model.raterEmbedding.weight.cpu().flatten().detach().numpy(),
      c.lowDiligenceRaterInterceptRound2Key: raterIntercept2,
    }
  )
  return noteStats, raterStats, globalIntercept
