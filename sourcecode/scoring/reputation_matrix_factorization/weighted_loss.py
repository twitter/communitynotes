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

  def __init__(
    self,
    loss_fn,
    notes,
    raters,
    targets,
    posWeight=1,
    raterReputationTensor=None,
    alpha=0,
    raterWeightPower=1,
    noteWeightPower=0,  # set to zero to weight according to ratings, -1 to weight all notes equally
    device=torch.device("cpu"),
  ):
    super().__init__()
    self.loss_fn = loss_fn
    self.targets = targets
    # calculate weights
    weights = torch.ones(targets.shape[0], device=device)
    # apply positive weight
    weights[targets == 1] = posWeight
    # apply rater reputation weight
    if raterReputationTensor is not None:
      raterReputation = nn.Embedding(raterReputationTensor.shape[0], 1)
      raterReputation.weight = nn.Parameter(raterReputationTensor)
      raterReputation.weight.requires_grad_(False)
      ratingWeights = raterReputation(raters).flatten() ** raterWeightPower
      ratingWeights = (alpha + ratingWeights) / (1 + alpha)
      weights = weights * ratingWeights
    # apply note weight normalization
    if noteWeightPower != 0:
      noteWeightTotals = self._get_note_weight_totals(notes, raters, weights)
      noteWeightTotals = noteWeightTotals.reshape(-1, 1).to(device)
      noteWeightTable = nn.Embedding(noteWeightTotals.shape[0], 1, device=device)
      noteWeightTable.weight = nn.Parameter(noteWeightTotals)
      noteWeightTable.weight.requires_grad_(False)
      noteWeights = noteWeightTable(notes).flatten()
      noteWeights = noteWeights**noteWeightPower
      weights = weights * noteWeights
    self.weights = weights

  def forward(self, pred):
    return (self.weights * self.loss_fn(pred, self.targets)).mean()
