from dataclasses import dataclass
import logging
from typing import Optional

import torch


logger = logging.getLogger("birdwatch.model")
logger.setLevel(logging.INFO)


@dataclass
class ModelData:
  rating_labels: Optional[torch.FloatTensor]
  user_indexes: Optional[torch.IntTensor]
  note_indexes: Optional[torch.IntTensor]


class BiasedMatrixFactorization(torch.nn.Module):
  """Matrix factorization algorithm class."""

  def __init__(
    self,
    n_users: int,
    n_notes: int,
    n_factors: int = 1,
    use_global_intercept: bool = True,
    log: bool = True,
    seed: Optional[int] = None,
  ) -> None:
    """Initialize matrix factorization model using xavier_uniform for factors
    and zeros for intercepts.

    Args:
        n_users (int): number of raters
        n_notes (int): number of notes
        n_factors (int, optional): number of dimensions. Defaults to 1. Only 1 is supported.
        use_global_intercept (bool, optional): Defaults to True.
    """
    super().__init__()

    self._log = log

    self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=False, dtype=torch.float32)
    self.note_factors = torch.nn.Embedding(n_notes, n_factors, sparse=False, dtype=torch.float32)

    self.user_intercepts = torch.nn.Embedding(n_users, 1, sparse=False, dtype=torch.float32)
    self.note_intercepts = torch.nn.Embedding(n_notes, 1, sparse=False, dtype=torch.float32)

    self.use_global_intercept = use_global_intercept
    self.global_intercept = torch.nn.parameter.Parameter(torch.zeros(1, 1, dtype=torch.float32))
    if seed is not None:
      torch.manual_seed(seed)
    torch.nn.init.xavier_uniform_(self.user_factors.weight)
    torch.nn.init.xavier_uniform_(self.note_factors.weight)
    self.user_intercepts.weight.data.fill_(0.0)
    self.note_intercepts.weight.data.fill_(0.0)
    self.device = torch.device(
      "cuda:0"
      if torch.cuda.is_available()
      else "mps"
      if torch.backends.mps.is_available()
      else "cpu"
    )

  def forward(self, data: ModelData):
    """
    Forward pass: get predicted rating for user of note

    Args: data: ModelData object with the following attributes:
        user_indexes (torch.LongTensor): user id, shape (batch_size,)
        note_indexes (torch.LongTensor): note id, shape (batch_size,)

    Returns:
        torch.FloatTensor: predicted rating, shape (batch_size,)
    """
    pred = self.user_intercepts(data.user_indexes) + self.note_intercepts(data.note_indexes)
    pred += (self.user_factors(data.user_indexes) * self.note_factors(data.note_indexes)).sum(
      1, keepdim=True
    )
    if self.use_global_intercept == True:
      pred += self.global_intercept
    return pred.squeeze()

  def freeze_rater_and_global_parameters(self):
    """Freeze rater and global parameters."""
    self._freeze_parameters({"user", "global"})

  def freeze_factors(self):
    """Freeze factors."""
    self._freeze_parameters({"factor"})

  def _freeze_parameters(self, words_to_freeze: set):
    """Freeze rater and global parameters."""
    for name, param in self.named_parameters():
      for word in words_to_freeze:
        if word in name:
          if self._log:
            logger.info(f"Freezing parameter: {name}")
          param.requires_grad_(False)
