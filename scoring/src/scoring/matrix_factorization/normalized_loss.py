from dataclasses import dataclass
from typing import Optional

from .. import constants as c

import torch


@dataclass
class NormalizedLossHyperparameters:
  globalSignNorm: bool
  noteSignAlpha: Optional[float]
  noteNormExp: float
  raterNormExp: float


class NormalizedLoss(torch.nn.Module):
  def _set_note_norm_weights(self, ratings, exponent):
    df = ratings[[c.noteIdKey, "weights"]]
    df = df.groupby(c.noteIdKey).sum().reset_index(drop=False)
    df["multiplier"] = df["weights"] ** exponent
    ratings = ratings.merge(df[[c.noteIdKey, "multiplier"]])
    ratings["weights"] = ratings["weights"] * ratings["multiplier"]
    return ratings.drop(columns="multiplier")

  def _set_rater_norm_weights(self, ratings, exponent):
    df = ratings[[c.raterParticipantIdKey, "weights"]]
    df = df.groupby(c.raterParticipantIdKey).sum().reset_index(drop=False)
    df["multiplier"] = df["weights"] ** exponent
    ratings = ratings.merge(df[[c.raterParticipantIdKey, "multiplier"]])
    ratings["weights"] = ratings["weights"] * ratings["multiplier"]
    return ratings.drop(columns="multiplier")

  def _set_global_sign_weights(self, ratings):
    origTotalWeight = ratings["weights"].sum()
    posTotal = (ratings[c.internalRaterFactor1Key] > 0).sum()
    negTotal = (ratings[c.internalRaterFactor1Key] <= 0).sum()
    posRatio = negTotal / posTotal
    ratings.loc[ratings[c.internalRaterFactor1Key] > 0, "weights"] = (
      posRatio * ratings[ratings[c.internalRaterFactor1Key] > 0]["weights"]
    )
    newTotalWeight = ratings["weights"].sum()
    ratings["weights"] = (origTotalWeight / newTotalWeight) * ratings["weights"]
    return ratings

  def _set_note_sign_weights(self, ratings, alpha):
    assert alpha is not None
    # Save total weight and total note weights for reference
    origNoteTotalWeight = (
      ratings[[c.noteIdKey, "weights"]]
      .groupby(c.noteIdKey)
      .sum()
      .reset_index(drop=False)
      .rename(columns={"weights": "origNoteTotal"})
    )
    origTotalWeight = origNoteTotalWeight["origNoteTotal"].values.sum()
    # Calculate positive rating weight updates
    notePosTotals = (
      ratings[ratings[c.internalRaterFactor1Key] > 0][[c.noteIdKey, "weights"]]
      .groupby(c.noteIdKey)
      .sum()
      .reset_index(drop=False)
    )
    notePosTotals = notePosTotals.rename(columns={"weights": "notePosTotal"})
    noteNegTotals = (
      ratings[ratings[c.internalRaterFactor1Key] <= 0][[c.noteIdKey, "weights"]]
      .groupby(c.noteIdKey)
      .sum()
      .reset_index(drop=False)
    )
    noteNegTotals = noteNegTotals.rename(columns={"weights": "noteNegTotal"})
    tmp = notePosTotals.merge(
      noteNegTotals
    )  # OK if we drop some notes - notes with only positive or negative ratings get no update
    tmp["multiplier"] = (alpha + tmp["noteNegTotal"]) / (alpha + tmp["notePosTotal"])
    # Apply positive rating weight updates
    tmp = ratings.merge(tmp[[c.noteIdKey, "multiplier"]], how="left").fillna({"multiplier": 1})
    tmp.loc[tmp[c.internalRaterFactor1Key] <= 0, "multiplier"] = 1
    tmp["weights"] = tmp["weights"] * tmp["multiplier"]
    # Renormalize totals allocated to each note
    newNoteTotalWeight = (
      tmp[[c.noteIdKey, "weights"]]
      .groupby(c.noteIdKey)
      .sum()
      .reset_index(drop=False)
      .rename(columns={"weights": "newNoteTotal"})
    )
    tmp = tmp.merge(newNoteTotalWeight)
    tmp["weights"] = tmp["weights"] * (tmp["newNoteTotal"] ** -1)
    tmp = tmp.merge(origNoteTotalWeight)
    tmp["weights"] = tmp["weights"] * tmp["origNoteTotal"]
    # Re-form ratings, validate and return
    ratings = tmp[[c.noteIdKey, c.raterParticipantIdKey, c.internalRaterFactor1Key, "weights"]]
    assert (
      abs(origTotalWeight - ratings["weights"].sum()) < 10**-5
    ), f"{origTotalWeight} vs {ratings['weights'].sum()}"
    return ratings

  def __init__(
    self, criterion, ratings, targets, hparams, labelCol, raterFactors=None, device="cpu"
  ):
    super().__init__()
    # Initialize members
    self.loss_fn = criterion
    self.targets = targets
    # Validate that ratings is ordered correctly and preserve order
    assert len(ratings) == len(targets)
    assert all(ratings[labelCol].values == targets.cpu().numpy())
    ratingOrder = ratings[[c.raterParticipantIdKey, c.noteIdKey]].copy()
    # Assign factors if applicable
    if raterFactors is not None:
      assert not any(raterFactors[c.internalRaterFactor1Key].isna())
      ratings = ratings.merge(raterFactors)
      assert len(ratings) == len(targets)
    self.targets = self.targets.to(device)
    # Calculate weights
    ratings["weights"] = 1.0
    if hparams.noteSignAlpha is not None:
      ratings = self._set_note_sign_weights(ratings, hparams.noteSignAlpha)
    if hparams.globalSignNorm:
      ratings = self._set_global_sign_weights(ratings)
    if hparams.noteNormExp:
      ratings = self._set_note_norm_weights(ratings, hparams.noteNormExp)
    if hparams.raterNormExp:
      ratings = self._set_rater_norm_weights(ratings, hparams.raterNormExp)
    # Finalize weights
    weightMap = dict(
      ((rater, note), weight)
      for (rater, note, weight) in zip(
        ratings[c.raterParticipantIdKey], ratings[c.noteIdKey], ratings["weights"]
      )
    )
    self.weights = torch.FloatTensor(
      [
        weightMap[(rater, note)]
        for (rater, note) in zip(ratingOrder[c.raterParticipantIdKey], ratingOrder[c.noteIdKey])
      ]
    )
    assert len(self.weights) == len(self.targets)
    self.weights = self.weights.to(device)

  def forward(self, pred):
    return (self.weights * self.loss_fn(pred, self.targets)).mean()
