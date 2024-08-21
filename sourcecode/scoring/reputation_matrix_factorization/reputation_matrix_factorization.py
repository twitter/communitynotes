from dataclasses import dataclass
import logging
import time
from typing import Optional

from .. import constants as c
from .weighted_loss import WeightedLoss

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


logger = logging.getLogger("birdwatch.reputation_matrix_factorization")
logger.setLevel(logging.INFO)


# Define dataclass to represent learning hyperparameters
@dataclass
class ReputationModelHyperparameters:
  # Model hyperparameters
  activationFunction: str
  nDim: int
  # Optimizaiton hyperparameters
  numEpochs: int
  logRate: int
  learningRate: float
  convergence: float
  stablePeriod: int
  # Regularization hyperparameters
  l2Lambda: float
  l2NoteBiasMultiplier: float
  l2RaterBiasMultiplier: float
  l2GlobalBiasMultiplier: float
  l2RaterReputationMultiplier: float
  l2LambdaThirdRoundMultiplier: float
  l2NoteBiasThirdRoundMultiplier: float
  # Base / first round loss hyperparameters
  lossFunction: str
  posWeight: float
  noteNormExpFirstRound: float
  raterNormExpFirstRound: float
  # Second round loss hyperparameters
  posWeightSecondRoundMultiplier: float
  noteNormExpSecondRound: float
  raterNormExpSecondRound: float
  # Third round loss hyperparameters
  posWeightThirdRoundMultiplier: float
  noteNormExpThirdRound: float
  raterNormExpThirdRound: float
  reputationExp: float
  alpha: float
  defaultReputation: float = 1.0
  ratingPerNoteLossRatio: Optional[float] = None
  ratingPerUserLossRatio: Optional[float] = None


def get_or_default_if_nan(lookupDict, key, default):
  if key not in lookupDict:
    return default
  val = lookupDict.get(key, default)
  if np.isnan(val):
    return default
  return val


# Define model with customizable loss, activation, regularization and dimensionality
class ReputationMFModel(nn.Module):
  def __init__(
    self,
    dataset,
    activation_fn,
    nDim,
    l2Lambda,
    l2NoteBiasMultiplier,
    l2RaterBiasMultiplier,
    l2GlobalBiasMultiplier,
    l2RaterReputationMultiplier,
    noteInitState: Optional[pd.DataFrame] = None,
    raterInitState: Optional[pd.DataFrame] = None,
    globalInterceptInit: Optional[float] = None,
    device=torch.device("cpu"),
    defaultReputation=1.0,
    ratingPerNoteLossRatio: Optional[float] = None,
    ratingPerUserLossRatio: Optional[float] = None,
  ):
    """
    noteInitState expects a df with columns:
      noteId, internalNoteIntercept, internalRaterFactor1
    raterInitState expects a df with columns:
      raterParticipantIdKey, internalRaterIntercept, internalRaterFactor1, internalRaterReputation

    For diligence model: may want to map these names back to internal before calling this function.
    """
    super().__init__()
    # Save hyperparameters
    self.activation_fn = activation_fn
    self.nDim = nDim
    self.l2Lambda = l2Lambda
    self.l2NoteBiasMultiplier = l2NoteBiasMultiplier
    self.l2RaterBiasMultiplier = l2RaterBiasMultiplier
    self.l2RaterReputationMultiplier = l2RaterReputationMultiplier
    self.l2GlobalBiasMultiplier = l2GlobalBiasMultiplier
    self.format = {"device": device, "dtype": torch.float32}
    # Create parameters
    self.noteEmbedding = nn.Embedding(dataset.notes.shape[0], self.nDim, **self.format)
    self.raterEmbedding = nn.Embedding(dataset.raters.shape[0], self.nDim, **self.format)
    self.noteBias = nn.Embedding(dataset.notes.shape[0], 1, **self.format)
    self.raterBias = nn.Embedding(dataset.raters.shape[0], 1, **self.format)
    self.raterReputation = nn.Embedding(dataset.raters.shape[0], 1, **self.format)
    self.raterReputation.weight = nn.Parameter(
      torch.ones(self.raterReputation.weight.shape[0], 1, **self.format) * defaultReputation
    )
    self._ratingPerNoteLossRatio = ratingPerNoteLossRatio
    self._ratingPerUserLossRatio = ratingPerUserLossRatio

    self.init_global_bias(globalInterceptInit)
    self.init_rater_factor(raterInitState, dataset, device, defaultValue=0.0)
    self.init_rater_intercept(raterInitState, dataset, device, defaultValue=0.0)
    self.init_rater_reputation(raterInitState, dataset, device, defaultValue=defaultReputation)
    self.init_note_factor(noteInitState, dataset, device, defaultValue=0.0)
    self.init_note_intercept(noteInitState, dataset, device, defaultValue=0.0)

  def init_parameter(self, initDf, initCol, idKey, ratersOrNotes, device, defaultValue):
    if initDf is not None and initCol in initDf.columns:
      idToInitValue = dict(initDf[[idKey, initCol]].values)
      logger.info(f"Initializing {initCol}:")
      logger.info(
        f"  num in dataset: {ratersOrNotes.shape[0]}, vs. num we are initializing: {len(initDf)}"
      )
      paramWeightToInit = nn.Parameter(
        torch.tensor(
          [
            get_or_default_if_nan(lookupDict=idToInitValue, key=raterOrNoteId, default=defaultValue)
            for raterOrNoteId in ratersOrNotes
          ]
        )
        .to(torch.float32)
        .reshape(-1, 1)
        .to(device)
      )
      logger.info(f"  uninitialized {initCol}s: {(paramWeightToInit == 0).flatten().sum()}")
      logger.info(f"  initialized {initCol}s: {(paramWeightToInit != 0).flatten().sum()}")
      return paramWeightToInit
    else:
      logger.info(f"Not initializing {initCol}")
      return None

  def init_note_factor(self, noteInitState, dataset, device, defaultValue=0):
    initVal = self.init_parameter(
      initDf=noteInitState,
      initCol=c.internalNoteFactor1Key,
      idKey=c.noteIdKey,
      ratersOrNotes=dataset.notes,
      device=device,
      defaultValue=defaultValue,
    )
    if initVal is not None:
      self.noteEmbedding.weight = initVal
    assert not torch.isnan(self.noteEmbedding.weight).any()

  def init_note_intercept(self, noteInitState, dataset, device, defaultValue=0):
    initVal = self.init_parameter(
      initDf=noteInitState,
      initCol=c.internalNoteInterceptKey,
      idKey=c.noteIdKey,
      ratersOrNotes=dataset.notes,
      device=device,
      defaultValue=defaultValue,
    )
    if initVal is not None:
      self.noteBias.weight = initVal
    assert not torch.isnan(self.noteBias.weight).any()

  def init_rater_factor(self, raterInitState, dataset, device, defaultValue=0):
    initVal = self.init_parameter(
      initDf=raterInitState,
      initCol=c.internalRaterFactor1Key,
      idKey=c.raterParticipantIdKey,
      ratersOrNotes=dataset.raters,
      device=device,
      defaultValue=defaultValue,
    )
    if initVal is not None:
      self.raterEmbedding.weight = initVal
    assert not torch.isnan(self.raterEmbedding.weight).any()

  def init_rater_reputation(self, raterInitState, dataset, device, defaultValue):
    initVal = self.init_parameter(
      initDf=raterInitState,
      initCol=c.internalRaterReputationKey,
      idKey=c.raterParticipantIdKey,
      ratersOrNotes=dataset.raters,
      device=device,
      defaultValue=defaultValue,
    )
    if initVal is not None:
      self.raterReputation.weight = initVal
    assert not torch.isnan(self.raterReputation.weight).any()

  def init_rater_intercept(self, raterInitState, dataset, device, defaultValue=0):
    initVal = self.init_parameter(
      initDf=raterInitState,
      initCol=c.internalRaterInterceptKey,
      idKey=c.raterParticipantIdKey,
      ratersOrNotes=dataset.raters,
      device=device,
      defaultValue=defaultValue,
    )
    if initVal is not None:
      self.raterBias.weight = initVal
    assert not torch.isnan(self.raterBias.weight).any()

  def init_global_bias(self, globalInterceptInit):
    if globalInterceptInit is not None:
      self.globalBias = nn.Parameter(torch.tensor(globalInterceptInit, **self.format))
    else:
      self.globalBias = nn.Parameter(torch.tensor(0.0, **self.format))
    assert not torch.isnan(self.globalBias).any()

  def forward(self, notes, raters):
    pred = (self.noteEmbedding(notes) * self.raterEmbedding(raters)).sum(
      axis=1, keepdim=True
    ) / np.sqrt(self.nDim)
    pred += self.noteBias(notes) * self.raterReputation(raters)
    pred += self.raterBias(raters) + self.globalBias
    return self.activation_fn(pred)

  def get_regularization_loss(self, numRatings):
    regularizationLoss = self.l2Lambda * self.l2GlobalBiasMultiplier * (self.globalBias**2)

    if self._ratingPerNoteLossRatio is None:
      regularizationLoss += self.l2Lambda * (self.noteEmbedding.weight**2).mean()
      regularizationLoss += (
        self.l2Lambda * self.l2NoteBiasMultiplier * (self.noteBias.weight**2).mean()
      )
    else:
      simulatedNumberOfNotesForLoss = numRatings / self._ratingPerNoteLossRatio
      regularizationLoss += (
        self.l2Lambda * (self.noteEmbedding.weight**2).sum() / simulatedNumberOfNotesForLoss
      )
      regularizationLoss += (
        self.l2Lambda
        * self.l2NoteBiasMultiplier
        * (self.noteBias.weight**2).sum()
        / simulatedNumberOfNotesForLoss
      )

    if self._ratingPerUserLossRatio is None:
      regularizationLoss += self.l2Lambda * (self.raterEmbedding.weight**2).mean()
      regularizationLoss += (
        self.l2Lambda * self.l2RaterBiasMultiplier * (self.raterBias.weight**2).mean()
      )
      regularizationLoss += (
        self.l2Lambda * self.l2RaterReputationMultiplier * (self.raterReputation.weight**2).mean()
      )
    else:
      simulatedNumberOfRatersForLoss = numRatings / self._ratingPerUserLossRatio
      regularizationLoss += (
        self.l2Lambda * (self.raterEmbedding.weight**2).sum() / simulatedNumberOfRatersForLoss
      )
      regularizationLoss += (
        self.l2Lambda
        * self.l2RaterBiasMultiplier
        * (self.raterBias.weight**2).sum()
        / simulatedNumberOfRatersForLoss
      )
      regularizationLoss += (
        self.l2Lambda
        * self.l2RaterReputationMultiplier
        * (self.raterReputation.weight**2).sum()
        / simulatedNumberOfRatersForLoss
      )

    return regularizationLoss


# Define single pass training loop
def _train_one_round(model, loss_fn, dataset, hParams):
  # Identify tensors for training and testing
  notes = dataset.noteTensor
  raters = dataset.raterTensor
  numRatings = dataset.raterTensor.shape[0]
  # Initilaize training state
  optim = torch.optim.Adam(model.parameters(), lr=hParams.learningRate)
  epoch = 0
  start = time.time()
  priorLoss = None
  while epoch <= hParams.numEpochs:
    # Set gradients to zero
    optim.zero_grad()
    # Perform forward pass
    pred = model(notes, raters)
    # Compute loss
    loss = loss_fn(pred.flatten())
    loss += model.get_regularization_loss(numRatings)
    assert not torch.isnan(loss).any()
    if hParams.logRate and epoch % hParams.logRate == 0:
      logger.info(f"epoch={epoch:03d} | loss={loss.item():7.6f} | time={time.time() - start:.1f}s")
    if hParams.convergence > 0 and epoch % hParams.stablePeriod == 0:
      if priorLoss is not None and (priorLoss - loss).abs() < hParams.convergence:
        if hParams.logRate:
          logger.info(
            f"epoch={epoch:03d} | loss={loss.item():7.6f} | time={time.time() - start:.1f}s"
          )
        break
      priorLoss = loss
    # Perform backward pass
    loss.backward()
    # Update parameters
    optim.step()
    # Increment epoch
    epoch += 1
  return loss.item()


def _sigmoid_range(low, high):
  sigmoid_fn = torch.nn.Sigmoid()
  return lambda tensor: sigmoid_fn(tensor) * (high - low) + low


def _setup_model(
  dataset,  # MatrixFactorizationDataset,
  hParams: ReputationModelHyperparameters,
  noteInitState: pd.DataFrame,
  raterInitState: pd.DataFrame,
  globalInterceptInit: Optional[float] = None,
):
  # Define model
  activation_fn = None
  if hParams.activationFunction == "SIGMOID":
    activation_fn = nn.Sigmoid()
  elif hParams.activationFunction == "SIGMOID_RANGE":
    activation_fn = _sigmoid_range(-0.2, 1.2)
  else:
    assert hParams.activationFunction == "IDENTITY"
    activation_fn = nn.Identity()
  loss_fn = None
  if hParams.lossFunction == "MSELoss":
    loss_fn = nn.MSELoss(reduction="none")
  else:
    assert hParams.lossFunction == "BCEWithLogitsLoss"
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

  logger.info(
    f"Setup model: noteInitState: \n{noteInitState},\n raterInitState: \n{raterInitState}"
  )
  model = ReputationMFModel(
    dataset,
    activation_fn=activation_fn,
    nDim=hParams.nDim,
    l2Lambda=hParams.l2Lambda,
    l2NoteBiasMultiplier=hParams.l2NoteBiasMultiplier,
    l2RaterBiasMultiplier=hParams.l2RaterBiasMultiplier,
    l2GlobalBiasMultiplier=hParams.l2GlobalBiasMultiplier,
    l2RaterReputationMultiplier=hParams.l2RaterReputationMultiplier,
    noteInitState=noteInitState,
    raterInitState=raterInitState,
    globalInterceptInit=globalInterceptInit,
    defaultReputation=hParams.defaultReputation,
    ratingPerNoteLossRatio=hParams.ratingPerNoteLossRatio,
    ratingPerUserLossRatio=hParams.ratingPerUserLossRatio,
  )
  return model, loss_fn


# TODO: replace string constants with enums
def train_model_prescoring(
  hParams,
  dataset,
  noteInitState: Optional[pd.DataFrame] = None,
  raterInitState: Optional[pd.DataFrame] = None,
  device=torch.device("cpu"),
):
  model, loss_fn = _setup_model(dataset, hParams, noteInitState, raterInitState)

  logger.info("Reputation Matrix Factorization: rater reputation frozen")
  logger.info("Round 1:")
  loss_fn_1 = WeightedLoss(
    loss_fn,
    dataset.noteTensor,
    dataset.raterTensor,
    dataset.targetTensor,
    posWeight=hParams.posWeight,
    noteNormExp=hParams.noteNormExpFirstRound,
    raterNormExp=hParams.raterNormExpFirstRound,
    device=device,
  )
  model.raterReputation.requires_grad_(False)
  loss1 = _train_one_round(model, loss_fn_1, dataset, hParams)
  logger.info(f"After round 1, global bias: {model.globalBias}")
  globalInt1 = model.globalBias.data.cpu().detach().numpy().item()

  logger.info("\nRound 2: learn rater rep (and everything else), freeze note intercept")
  loss_fn_2 = WeightedLoss(
    loss_fn,
    dataset.noteTensor,
    dataset.raterTensor,
    dataset.targetTensor,
    posWeight=hParams.posWeight * hParams.posWeightSecondRoundMultiplier,
    noteNormExp=hParams.noteNormExpSecondRound,
    raterNormExp=hParams.raterNormExpSecondRound,
    device=device,
  )
  model.raterReputation.requires_grad_(True)
  model.noteBias.requires_grad_(False)

  loss2 = _train_one_round(model, loss_fn_2, dataset, hParams)
  globalInt2 = model.globalBias.data.cpu().detach().numpy().item()
  noteIntercept2 = model.noteBias.weight.cpu().flatten().detach().numpy().copy()
  raterIntercept2 = model.raterBias.weight.cpu().flatten().detach().numpy().copy()

  logger.info("\nRound 3: fit intercepts and global intercept with everything else frozen")
  model.l2Lambda = hParams.l2Lambda * hParams.l2LambdaThirdRoundMultiplier
  model.l2NoteBiasMultiplier = hParams.l2NoteBiasMultiplier * hParams.l2NoteBiasThirdRoundMultiplier
  model.noteBias.requires_grad_(True)
  model.noteEmbedding.requires_grad_(False)
  model.raterEmbedding.requires_grad_(False)
  model.raterReputation.requires_grad_(False)

  raterReputation = model.raterReputation.weight.detach().clone().clip(min=0)
  loss_fn_3 = WeightedLoss(
    loss_fn,
    dataset.noteTensor,
    dataset.raterTensor,
    dataset.targetTensor,
    posWeight=hParams.posWeight * hParams.posWeightThirdRoundMultiplier,
    raterReputation=raterReputation,
    reputationExp=hParams.reputationExp,
    alpha=hParams.alpha,
    noteNormExp=hParams.noteNormExpThirdRound,
    raterNormExp=hParams.raterNormExpThirdRound,
    device=device,
  )

  loss3 = _train_one_round(model, loss_fn_3, dataset, hParams)

  logger.info(f"After round 3, global bias: {model.globalBias}")
  globalInt3 = model.globalBias.data.cpu().detach().numpy().item()
  globalIntercept = c.ReputationGlobalIntercept(
    firstRound=globalInt1, secondRound=globalInt2, finalRound=globalInt3
  )

  return model, loss1, loss2, loss3, globalIntercept, noteIntercept2, raterIntercept2


def train_model_final(
  hParams,
  dataset,
  noteInitState: pd.DataFrame,
  raterInitState: pd.DataFrame,
  globalInterceptInit: c.ReputationGlobalIntercept,
  device=torch.device("cpu"),
):
  """
  Args:
    hParams (ReputationModelHyperparameters)
    dataset (ReputationDataset)
    noteInitState (Optional[pd.DataFrame]): expects internal column names e.g. internalNoteIntercept
    raterInitState (Optional[pd.DataFrame]): expects internal column names e.g. internalRaterIntercept
  """
  hParams.defaultReputation = 0.0  # 0 reputation for raters missing from init.

  # setup_model initializes uses the internal intercepts, but we want to initialize with round 2 intercepts,
  #  and save the final rater intercepts for later initialization.
  noteInitState[c.internalNoteInterceptKey] = noteInitState[c.internalNoteInterceptRound2Key]

  savedFinalRoundPrescoringRaterIntercept = raterInitState[c.internalRaterInterceptKey].copy()
  raterInitState[c.internalRaterInterceptKey] = raterInitState[c.internalRaterInterceptRound2Key]

  model, loss_fn = _setup_model(
    dataset, hParams, noteInitState, raterInitState, globalInterceptInit.secondRound
  )

  logger.info(
    "Final scoring, initial round fitting reputation MF (equivalent to Round 2 in Prescoring - learn note factor)"
  )

  model.noteBias.requires_grad_(False)
  model.noteEmbedding.requires_grad_(True)
  model.raterEmbedding.requires_grad_(False)
  model.raterReputation.requires_grad_(False)
  model.raterBias.requires_grad_(False)
  model.globalBias.requires_grad_(False)

  loss_fn_2 = WeightedLoss(
    loss_fn,
    dataset.noteTensor,
    dataset.raterTensor,
    dataset.targetTensor,
    posWeight=hParams.posWeight * hParams.posWeightSecondRoundMultiplier,
    noteNormExp=hParams.noteNormExpSecondRound,
    raterNormExp=hParams.raterNormExpSecondRound,
    device=device,
  )
  _train_one_round(model, loss_fn_2, dataset, hParams)

  logger.info("Final scoring, final round fitting reputation MF: learn just note intercept")

  # Now set the global intercept to the value from the final round
  model.globalBias.data = torch.tensor(globalInterceptInit.finalRound, **model.format)

  # Set rater intercepts back to final round. We will learn note intercepts, so no need to set them back.
  raterInitState[c.internalRaterInterceptKey] = savedFinalRoundPrescoringRaterIntercept
  model.init_rater_intercept(raterInitState, dataset, device)

  model.l2Lambda = hParams.l2Lambda * hParams.l2LambdaThirdRoundMultiplier
  model.l2NoteBiasMultiplier = hParams.l2NoteBiasMultiplier * hParams.l2NoteBiasThirdRoundMultiplier

  model.noteBias.requires_grad_(True)
  model.noteEmbedding.requires_grad_(False)
  model.raterEmbedding.requires_grad_(False)
  model.raterBias.requires_grad_(False)
  model.raterReputation.requires_grad_(False)
  model.globalBias.requires_grad_(False)

  raterReputation = model.raterReputation.weight.detach().clone().clip(min=0)
  loss_fn_final = WeightedLoss(
    loss_fn,
    dataset.noteTensor,
    dataset.raterTensor,
    dataset.targetTensor,
    posWeight=hParams.posWeight * hParams.posWeightThirdRoundMultiplier,
    raterReputation=raterReputation,
    reputationExp=hParams.reputationExp,
    alpha=hParams.alpha,
    noteNormExp=hParams.noteNormExpThirdRound,
    raterNormExp=hParams.raterNormExpThirdRound,
    device=device,
  )

  loss_final = _train_one_round(model, loss_fn_final, dataset, hParams)
  return model, loss_final
