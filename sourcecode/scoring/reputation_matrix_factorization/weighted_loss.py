import scipy.sparse as sps
import torch
import torch.nn as nn


# Define loss wrapper for weight support
class WeightedLoss(nn.Module):
  def _get_note_weight_totals(self, notes, raters, weights):
    # Move tensors to CPU due to MPS limitations with int64.
    notes = notes.cpu().detach().numpy()
    raters = raters.cpu().detach().numpy()
    weights = weights.cpu().detach().numpy()
    # Guarantee a weight minimum for each note incase all ratings have been assigned zero
    # weight.  This prevents divide by zero errors during normalization.
    return torch.from_numpy(sps.coo_matrix((weights, (notes, raters))).sum(axis=1) + 2**-20)

  def _get_rater_weight_totals(self, notes, raters, weights):
    # Move tensors to CPU due to MPS limitations with int64.
    notes = notes.cpu().detach().numpy()
    raters = raters.cpu().detach().numpy()
    weights = weights.cpu().detach().numpy()
    # Guarantee a weight minimum for each note incase all ratings have been assigned zero
    # weight.  This prevents divide by zero errors during normalization.
    return torch.from_numpy(sps.coo_matrix((weights, (notes, raters))).sum(axis=0) + 2**-20)

  def __init__(
    self,
    loss_fn,
    notes,
    raters,
    targets,
    posWeight=1,
    raterReputation=None,
    reputationExp=1,
    alpha=0,
    noteNormExp=0,
    raterNormExp=0,
    device=torch.device("cpu"),
  ):
    """Compute loss weighted by class, number of ratings per note and rater and reputation.

    Note that (as available) rater reputation is applied first, then note normalization, then rater normalization.

    Select Args:
    raterReputation: use only in 3rd pass after reputations are available
    reputationExp: set to 0 to remove effects of reputation, 0.5 to diminish, and 2 to exaggerate
    alpha: set >0 to introduce smoothing
    noteNormExp: set to zero to weight according to ratings, -1 to weight all notes equally, -0.5 to weight according to sqrt of total
    raterNormExp: set to zero to weight according to ratings, -1 to weight all raters equally, -0.5 to weight according to sqrt of total
    """
    super().__init__()
    self.loss_fn = loss_fn
    self.targets = targets
    # calculate weights
    weights = torch.ones(targets.shape[0], device=device)
    # apply positive weight
    weights[targets == 1] = posWeight
    # apply rater reputation weight
    if raterReputation is not None:
      raterReputationTable = nn.Embedding(raterReputation.shape[0], 1)
      raterReputationTable.weight = nn.Parameter(raterReputation)
      raterReputationTable.weight.requires_grad_(False)
      ratingWeights = raterReputationTable(raters).flatten() ** reputationExp
      ratingWeights = (alpha + ratingWeights) / (1 + alpha)
      weights = weights * ratingWeights
    # apply note weight normalization
    if noteNormExp != 0:
      noteWeightTotals = self._get_note_weight_totals(notes, raters, weights)
      noteWeightTotals = noteWeightTotals.reshape(-1, 1).to(device)
      noteWeightTable = nn.Embedding(noteWeightTotals.shape[0], 1, device=device)
      noteWeightTable.weight = nn.Parameter(noteWeightTotals)
      noteWeightTable.weight.requires_grad_(False)
      noteWeights = noteWeightTable(notes).flatten()
      noteWeights = noteWeights**noteNormExp
      weights = weights * noteWeights
    # apply rating normalization
    if raterNormExp != 0:
      raterWeightTotals = self._get_rater_weight_totals(notes, raters, weights)
      raterWeightTotals = raterWeightTotals.reshape(-1, 1).to(device)
      raterWeightTable = nn.Embedding(raterWeightTotals.shape[0], 1, device=device)
      raterWeightTable.weight = nn.Parameter(raterWeightTotals)
      raterWeightTable.weight.requires_grad_(False)
      raterWeights = raterWeightTable(raters).flatten()
      raterWeights = raterWeights**raterNormExp
      weights = weights * raterWeights
    self.weights = weights

  def forward(self, pred):
    return (self.weights * self.loss_fn(pred, self.targets)).mean()
