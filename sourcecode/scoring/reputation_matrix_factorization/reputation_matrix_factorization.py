import time

from .. import constants as c
from .weighted_loss import WeightedLoss

import numpy as np
import torch
import torch.nn as nn


# Define model with customizable loss, activation, regularization and dimensionality
class ReputationMFModel(nn.Module):
  def __init__(
    self,
    noteParams,
    raterParams,
    activation_fn=nn.Identity(),
    nDim=1,
    l2Lambda=0.03,
    l2NoteBiasMultiplier=5,
    l2RaterBiasMultiplier=5,
    l2RaterReputationMultiplier=5,
    l2GlobalBiasMultiplier=1,
    device=torch.device("cpu"),
    stableInit=False,
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
    self.noteEmbedding = nn.Embedding(len(noteParams), self.nDim, **self.format)
    self.raterEmbedding = nn.Embedding(len(raterParams), self.nDim, **self.format)
    self.noteBias = nn.Embedding(len(noteParams), 1, **self.format)
    self.raterBias = nn.Embedding(len(raterParams), 1, **self.format)
    self.raterReputation = nn.Embedding(len(raterParams), 1, **self.format)
    self.globalBias = nn.Parameter(torch.tensor(0.0, **self.format))
    # Initialize rater reputation to 1
    self.raterReputation.weight = nn.Parameter(
      torch.ones(self.raterReputation.weight.shape[0], 1, **self.format)
    )
    if stableInit:
      self.raterEmbedding.weight = nn.Parameter(
        torch.tensor(raterParams[c.internalRaterFactor1Key].values).reshape(-1, 1).to(device)
      )

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
def _train_one_round(
  model,
  loss_fn,
  dataset,
  numEpochs=150,
  learningRate=0.2,
  logRate=15,
  device=torch.device("cpu"),
  convergence=10**-7,
):
  # Identify tensors for training and testing
  notes, raters, _ = dataset
  # Initilaize training state
  optim = torch.optim.Adam(model.parameters(), lr=learningRate)
  epoch = 0
  start = time.time()
  priorLoss = None
  while epoch <= numEpochs:
    # Set gradients to zero
    optim.zero_grad()
    # Perform forward pass
    pred = model(notes, raters)
    # Compute loss
    loss = loss_fn(pred.flatten())
    loss += model.get_regularization_loss()
    if logRate and epoch % logRate == 0:
      print(f"epoch={epoch:03d} | loss={loss.item():7.4f} | time={time.time() - start:.1f}s")
    if priorLoss is not None and (priorLoss - loss).abs() < convergence:
      if logRate:
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


def train_model(
  model,
  dataset,
  loss_fn,
  posWeight=50,
  l2LambdaThirdRoundMultiplier=1,
  posWeightThirdRoundMultiplier=1,
  numEpochs=150,
  learningRate=0.2,
  logRate=15,
  device=torch.device("cpu"),
  alpha=0,
  raterWeightPower=1,
  initialNoteWeightPower=0,
  finalNoteWeightPower=-0.5,
  freezeRaterBias=False,
  resetReputation=False,
):
  # unpack dataset
  notes, raters, targets = dataset

  # train round 0
  print("Reputation Matrix Factorization:")
  print("Round 0:")
  initial_loss_fn = WeightedLoss(
    loss_fn,
    notes,
    raters,
    targets,
    posWeight=posWeight,
    noteNormExp=initialNoteWeightPower,
    device=device,
  )
  model.raterReputation.requires_grad_(False)
  loss0 = _train_one_round(
    model,
    initial_loss_fn,
    dataset,
    numEpochs=numEpochs,
    learningRate=learningRate,
    logRate=logRate,
    device=device,
    convergence=-1,
  )

  # train round 1
  print("\nRound 1:")
  model.raterReputation.requires_grad_(True)
  model.noteBias.requires_grad_(False)
  loss1 = _train_one_round(
    model,
    initial_loss_fn,
    dataset,
    numEpochs=numEpochs,
    learningRate=learningRate,
    logRate=logRate,
    device=device,
    convergence=-1,
  )

  # train round 2
  print("\nRound 2:")
  model.l2Lambda = model.l2Lambda * l2LambdaThirdRoundMultiplier
  model.noteBias.requires_grad_(True)
  model.noteEmbedding.requires_grad_(False)
  model.raterEmbedding.requires_grad_(False)
  model.raterReputation.requires_grad_(False)
  if freezeRaterBias:
    model.raterBias.requires_grad_(False)
  if resetReputation:
    model.reset_reputation()
  raterReputation = model.raterReputation.weight.detach().clone().clip(min=0)
  final_loss_fn = WeightedLoss(
    loss_fn,
    notes,
    raters,
    targets,
    posWeight=posWeight * posWeightThirdRoundMultiplier,
    raterReputation=raterReputation,
    reputationExp=raterWeightPower,
    alpha=alpha,
    noteNormExp=finalNoteWeightPower,
    device=device,
  )
  loss2 = _train_one_round(
    model,
    final_loss_fn,
    dataset,
    numEpochs=numEpochs,
    learningRate=learningRate,
    logRate=logRate,
    device=device,
    convergence=-1,
  )
  return loss0, loss1, loss2
