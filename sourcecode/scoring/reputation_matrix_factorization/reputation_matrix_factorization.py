from dataclasses import dataclass
import time
from typing import Optional

from .. import constants as c
from .weighted_loss import WeightedLoss

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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
    device=torch.device("cpu"),
  ):
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
    self.globalBias = nn.Parameter(torch.tensor(0.0, **self.format))
    # Initialize rater reputation to 1
    self.raterReputation.weight = nn.Parameter(
      torch.ones(self.raterReputation.weight.shape[0], 1, **self.format)
    )
    if raterInitState is not None:
      mapping = dict(raterInitState[[c.raterParticipantIdKey, c.internalRaterFactor1Key]].values)
      print("Initializing raters:")
      print(f"  num raters: {dataset.raters.shape[0]}")
      self.raterEmbedding.weight = nn.Parameter(
        torch.tensor([mapping.get(rater, 0.0) for rater in dataset.raters])
        .to(torch.float32)
        .reshape(-1, 1)
        .to(device)
      )
      print(f"  uninitialized raters: {(self.raterEmbedding.weight == 0).flatten().sum()}")
      print(f"  initialized raters: {(self.raterEmbedding.weight != 0).flatten().sum()}")
    if noteInitState is not None:
      print("Initializing notes:")
      print(f"  num notes: {dataset.notes.shape[0]}")
      mapping = dict(noteInitState[[c.noteIdKey, c.internalNoteFactor1Key]].values)
      self.noteEmbedding.weight = nn.Parameter(
        torch.tensor([mapping.get(note, 0.0) for note in dataset.notes])
        .reshape(-1, 1)
        .to(torch.float32)
        .to(device)
      )
      print(f"  uninitialized notes: {(self.noteEmbedding.weight == 0).flatten().sum()}")
      print(f"  initialized notes: {(self.noteEmbedding.weight != 0).flatten().sum()}")

  def forward(self, notes, raters):
    pred = (self.noteEmbedding(notes) * self.raterEmbedding(raters)).sum(
      axis=1, keepdim=True
    ) / np.sqrt(self.nDim)
    pred += self.noteBias(notes) * self.raterReputation(raters)
    pred += self.raterBias(raters) + self.globalBias
    return self.activation_fn(pred)

  def get_regularization_loss(self):
    regularizationLoss = (
      (self.l2Lambda * (self.noteEmbedding.weight**2).mean())
      + (self.l2Lambda * (self.raterEmbedding.weight**2).mean())
      + (self.l2Lambda * self.l2NoteBiasMultiplier * (self.noteBias.weight**2).mean())
      + (self.l2Lambda * self.l2RaterBiasMultiplier * (self.raterBias.weight**2).mean())
      + (self.l2Lambda * self.l2RaterReputationMultiplier * (self.raterReputation.weight**2).mean())
      + (self.l2Lambda * self.l2GlobalBiasMultiplier * (self.globalBias**2))
    )
    return regularizationLoss


# Define single pass training loop
def _train_one_round(model, loss_fn, dataset, hParams):
  # Identify tensors for training and testing
  notes = dataset.noteTensor
  raters = dataset.raterTensor
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
    loss += model.get_regularization_loss()
    if hParams.logRate and epoch % hParams.logRate == 0:
      print(f"epoch={epoch:03d} | loss={loss.item():7.4f} | time={time.time() - start:.1f}s")
    if hParams.convergence > 0 and epoch % hParams.stablePeriod == 0:
      if priorLoss is not None and (priorLoss - loss).abs() < hParams.convergence:
        if hParams.logRate:
          print(f"epoch={epoch:03d} | loss={loss.item():7.4f} | time={time.time() - start:.1f}s")
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


# TODO: replace string constants with enums
def train_model(
  hParams,
  dataset,
  noteInitState: Optional[pd.DataFrame] = None,
  raterInitState: Optional[pd.DataFrame] = None,
  device=torch.device("cpu"),
):
  # Unpack dataset
  notes = dataset.noteTensor
  raters = dataset.raterTensor
  targets = dataset.targetTensor

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
  )

  # train round 1
  print("Reputation Matrix Factorization:")
  print("Round 1:")
  loss_fn_1 = WeightedLoss(
    loss_fn,
    notes,
    raters,
    targets,
    posWeight=hParams.posWeight,
    noteNormExp=hParams.noteNormExpFirstRound,
    raterNormExp=hParams.raterNormExpFirstRound,
    device=device,
  )
  model.raterReputation.requires_grad_(False)
  loss1 = _train_one_round(model, loss_fn_1, dataset, hParams)

  # train round 2
  print("\nRound 2:")
  loss_fn_2 = WeightedLoss(
    loss_fn,
    notes,
    raters,
    targets,
    posWeight=hParams.posWeight * hParams.posWeightSecondRoundMultiplier,
    noteNormExp=hParams.noteNormExpSecondRound,
    raterNormExp=hParams.raterNormExpSecondRound,
    device=device,
  )
  model.raterReputation.requires_grad_(True)
  model.noteBias.requires_grad_(False)
  loss2 = _train_one_round(model, loss_fn_2, dataset, hParams)

  # train round 3
  print("\nRound 3:")
  model.l2Lambda = hParams.l2Lambda * hParams.l2LambdaThirdRoundMultiplier
  model.l2NoteBiasMultiplier = hParams.l2NoteBiasMultiplier * hParams.l2NoteBiasThirdRoundMultiplier
  model.noteBias.requires_grad_(True)
  model.noteEmbedding.requires_grad_(False)
  model.raterEmbedding.requires_grad_(False)
  model.raterReputation.requires_grad_(False)
  raterReputation = model.raterReputation.weight.detach().clone().clip(min=0)
  loss_fn_3 = WeightedLoss(
    loss_fn,
    notes,
    raters,
    targets,
    posWeight=hParams.posWeight * hParams.posWeightThirdRoundMultiplier,
    raterReputation=raterReputation,
    reputationExp=hParams.reputationExp,
    alpha=hParams.alpha,
    noteNormExp=hParams.noteNormExpThirdRound,
    raterNormExp=hParams.raterNormExpThirdRound,
    device=device,
  )
  loss3 = _train_one_round(model, loss_fn_3, dataset, hParams)

  return model, loss1, loss2, loss3
