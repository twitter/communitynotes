from typing import Optional, Tuple

import constants as c

import numpy as np
import pandas as pd
import torch


class BiasedMatrixFactorization(torch.nn.Module):
  """Matrix factorization algorithm class."""

  def __init__(
    self, n_users: int, n_items: int, n_factors: int = 1, use_global_intercept: bool = True
  ) -> None:
    """Initialize matrix factorization model using xavier_uniform for factors
    and zeros for intercepts.

    Args:
        n_users (int): number of raters
        n_items (int): number of notes
        n_factors (int, optional): number of dimensions. Defaults to 1. Only 1 is supported.
        use_global_intercept (bool, optional): Defaults to True.
    """
    super().__init__()
    self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=False)
    self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=False)
    self.user_intercepts = torch.nn.Embedding(n_users, 1, sparse=False)
    self.item_intercepts = torch.nn.Embedding(n_items, 1, sparse=False)
    self.use_global_intercept = use_global_intercept
    self.global_intercept = torch.nn.parameter.Parameter(torch.zeros(1, 1))
    torch.nn.init.xavier_uniform_(self.user_factors.weight)
    torch.nn.init.xavier_uniform_(self.item_factors.weight)
    self.user_intercepts.weight.data.fill_(0.0)
    self.item_intercepts.weight.data.fill_(0.0)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def forward(self, user, item):
    """Forward pass: get predicted rating for user of note (item)"""
    pred = self.user_intercepts(user) + self.item_intercepts(item)
    pred += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
    if self.use_global_intercept == True:
      pred += self.global_intercept
    return pred.squeeze()

  def freeze_rater_and_global_parameters(self):
    self.user_factors.requires_grad = False
    self.user_intercepts.requires_grad = False
    self.global_intercept.requires_grad = False


def get_note_and_rater_id_maps(
  ratings: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Args:
      ratings (pd.DataFrame)

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
  """
  # We are extracting only the subset of note data from the ratings data frame that is needed to
  # run matrix factorization. This avoids accidentally losing data through `dropna`.
  noteData = ratings[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]]
  assert not pd.isna(noteData).values.any(), "noteData must not contain nan values"

  raterIdMap = (
    pd.DataFrame(noteData[c.raterParticipantIdKey].unique())
    .reset_index()
    .set_index(0)
    .reset_index()
    .rename(columns={0: c.raterParticipantIdKey, "index": c.raterIndexKey})
  )

  noteIdMap = (
    pd.DataFrame(noteData[c.noteIdKey].unique())
    .reset_index()
    .set_index(0)
    .reset_index()
    .rename(columns={0: c.noteIdKey, "index": c.noteIndexKey})
  )

  noteRatingIds = noteData.merge(noteIdMap, on=c.noteIdKey)
  noteRatingIds = noteRatingIds.merge(raterIdMap, on=c.raterParticipantIdKey)

  return noteIdMap, raterIdMap, noteRatingIds


def initialize_parameters(
  mf_model: BiasedMatrixFactorization,
  noteIdMap: pd.DataFrame,
  raterIdMap: pd.DataFrame,
  noteInit: Optional[pd.DataFrame] = None,
  userInit: Optional[pd.DataFrame] = None,
  globalInterceptInit: Optional[float] = None,
  numFactors: int = 1,
) -> None:
  """If noteInit and userInit exist, use those to initialize note and user parameters.

  Args:
      mf_model (BiasedMatrixFactorization)
      noteIdMap (pd.DataFrame)
      raterIdMap (pd.DataFrame)
      noteInit (pd.DataFrame, optional)
      userInit (pd.DataFrame, optional)
      globalInterceptInit (float, optional)
  """
  if noteInit is not None:
    print("initializing notes")
    noteInit = noteIdMap.merge(noteInit, on=c.noteIdKey, how="left")
    mf_model.item_intercepts.weight.data = torch.tensor(
      np.expand_dims(noteInit[c.noteInterceptKey].astype(np.float32).values, axis=1)
    )
    mf_model.item_factors.weight.data = torch.tensor(
      noteInit[[c.note_factor_key(i) for i in range(1, numFactors + 1)]].astype(np.float32).values
    )

  if userInit is not None:
    print("initializing users")
    userInit = raterIdMap.merge(userInit, on=c.raterParticipantIdKey, how="left")
    mf_model.user_intercepts.weight.data = torch.tensor(
      np.expand_dims(userInit[c.raterInterceptKey].astype(np.float32).values, axis=1)
    )
    mf_model.user_factors.weight.data = torch.tensor(
      userInit[[c.rater_factor_key(i) for i in range(1, numFactors + 1)]].astype(np.float32).values
    )

  if globalInterceptInit is not None:
    print("initialized global intercept")
    mf_model.global_intercept = torch.nn.parameter.Parameter(torch.ones(1, 1) * globalInterceptInit)


def get_parameters_from_trained_model(
  mf_model: BiasedMatrixFactorization,
  noteIdMap: pd.DataFrame,
  raterIdMap: pd.DataFrame,
  flipFactorsForIdentification: bool = True,
  numFactors: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Args:
      mf_model (BiasedMatrixFactorization)
      noteIdMap (pd.DataFrame)
      raterIdMap (pd.DataFrame)
      flipFactorsForIdentification (bool, optional)

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: noteIdMap, raterIdMap
  """
  noteParams = noteIdMap.copy(deep=True)
  raterParams = raterIdMap.copy(deep=True)

  noteParams[c.noteInterceptKey] = mf_model.item_intercepts.weight.data.cpu().numpy()
  raterParams[c.raterInterceptKey] = mf_model.user_intercepts.weight.data.cpu().numpy()

  for i in range(numFactors):
    noteParams[c.note_factor_key(i + 1)] = mf_model.item_factors.weight.data.cpu().numpy()[:, i]
    raterParams[c.rater_factor_key(i + 1)] = mf_model.user_factors.weight.data.cpu().numpy()[:, i]

  if flipFactorsForIdentification:
    noteParams, raterParams = flip_factors_for_identification(noteParams, raterParams, numFactors)

  return noteParams, raterParams


def create_mf_model(
  noteIdMap: pd.DataFrame,
  raterIdMap: pd.DataFrame,
  noteRatingIds: pd.DataFrame,
  useGlobalIntercept: bool,
  numFactors: int,
  logging: bool = True,
  noteInit: Optional[pd.DataFrame] = None,
  userInit: Optional[pd.DataFrame] = None,
  globalInterceptInit: Optional[float] = None,
) -> Tuple[BiasedMatrixFactorization, torch.optim.Optimizer, torch.nn.modules.loss._Loss]:
  """Initialize BiasedMatrixFactorization model, optimizer, and loss.

  Args:
      noteIdMap (pd.DataFrame)
      raterIdMap (pd.DataFrame)
      noteRatingIds (pd.DataFrame)
      useGlobalIntercept (bool)
      numFactors (int)
      logging (bool, optional)
      noteInit (pd.DataFrame, optional)
      userInit (pd.DataFrame, optional)
      globalIntercept (float, optional)

  Returns:
      Tuple[BiasedMatrixFactorization, torch.optim.Optimizer, torch.nn.modules.loss._Loss]
  """
  n_users = noteRatingIds[c.raterIndexKey].nunique()
  n_items = noteRatingIds[c.noteIndexKey].nunique()
  if logging:
    print("------------------")
    print(f"Users: {n_users}, Notes: {n_items}")

  criterion = torch.nn.MSELoss()

  mf_model = BiasedMatrixFactorization(
    n_users, n_items, use_global_intercept=useGlobalIntercept, n_factors=numFactors
  )

  initialize_parameters(
    mf_model, noteIdMap, raterIdMap, noteInit, userInit, globalInterceptInit, numFactors
  )

  if (noteInit is not None) and (userInit is not None):
    optimizer = torch.optim.Adam(
      mf_model.parameters(), lr=c.initLearningRate
    )  # smaller learning rate
  else:
    optimizer = torch.optim.Adam(mf_model.parameters(), lr=c.noInitLearningRate)  # learning rate

  print(mf_model.device)
  mf_model.to(mf_model.device)

  return mf_model, optimizer, criterion


def fit_model(
  mf_model: BiasedMatrixFactorization,
  optimizer: torch.optim.Optimizer,
  criterion: torch.nn.modules.loss._Loss,
  l2_lambda: float,
  l2_intercept_multiplier: float,
  row: torch.LongTensor,
  col: torch.LongTensor,
  rating: torch.FloatTensor,
  logging: bool = True,
) -> None:
  """Run gradient descent to train the model.

  Args:
      mf_model (BiasedMatrixFactorization)
      optimizer (torch.optim.Optimizer)
      criterion (torch.nn.modules.loss._Loss)
      l2_lambda (float)
      l2_intercept_multiplier (float)
      row (torch.LongTensor)
      col (torch.LongTensor)
      rating (torch.FloatTensor)
      logging (bool, optional)
  """
  l2_lambda_intercept = l2_lambda * l2_intercept_multiplier

  def print_loss():
    y_pred = mf_model(row, col)
    train_loss = criterion(y_pred, rating)

    if logging:
      print("epoch", epoch, loss.item())
      print("TRAIN FIT LOSS: ", train_loss.item())

  prev_loss = 1e10

  y_pred = mf_model(row, col)
  loss = criterion(y_pred, rating)
  l2_reg_loss = torch.tensor(0.0).to(mf_model.device)

  for name, param in mf_model.named_parameters():
    if "intercept" in name:
      l2_reg_loss += l2_lambda_intercept * (param**2).mean()
    else:
      l2_reg_loss += l2_lambda * (param**2).mean()

  loss += l2_reg_loss

  epoch = 0

  while abs(prev_loss - loss.item()) > c.convergence:

    prev_loss = loss.item()

    # Backpropagate
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Set gradients to zero
    optimizer.zero_grad()

    # Predict and calculate loss
    y_pred = mf_model(row, col)
    loss = criterion(y_pred, rating)
    l2_reg_loss = torch.tensor(0.0).to(mf_model.device)

    for name, param in mf_model.named_parameters():
      if "intercept" in name:
        l2_reg_loss += l2_lambda_intercept * (param**2).mean()
      else:
        l2_reg_loss += l2_lambda * (param**2).mean()

    loss += l2_reg_loss

    if epoch % 50 == 0:
      print_loss()

    epoch += 1

  if logging:
    print("Num epochs:", epoch)
    print_loss()


def run_mf(
  ratings: pd.DataFrame,
  l2_lambda: float,
  l2_intercept_multiplier: float,
  numFactors: int,
  epochs: int,
  useGlobalIntercept: bool,
  runName: str = "prod",
  logging: bool = True,
  flipFactorsForIdentification: bool = True,
  noteInit: pd.DataFrame = None,
  userInit: pd.DataFrame = None,
  globalInterceptInit: Optional[float] = None,
  specificNoteId: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
  """Train matrix factorization model.

  See https://twitter.github.io/communitynotes/ranking-notes/#matrix-factorization

  Args:
      ratings (pd.DataFrame): pre-filtered ratings to train on
      l2_lambda (float): regularization for factors
      l2_intercept_multiplier (float): how much extra to regularize intercepts
      numFactors (int): number of dimensions (only 1 is implemented)
      epochs (int): number of rounds of training
      useGlobalIntercept (bool): whether to fit global intercept parameter
      runName (str, optional): name. Defaults to "prod".
      logging (bool, optional): debug output. Defaults to True.
      flipFactorsForIdentification (bool, optional): Default to True.
      noteInit (pd.DataFrame, optional)
      userInit (pd.DataFrame, optional)
      globalInterceptInit (float, optional)
      specificNoteId (int, optional)

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, float]:
        noteParams: contains one row per note, including noteId and learned note parameters
        raterParams: contains one row per rating, including raterId and learned rater parameters
        globalIntercept: learned global intercept parameter
  """
  noteIdMap, raterIdMap, noteRatingIds = get_note_and_rater_id_maps(ratings)

  mf_model, optimizer, criterion = create_mf_model(
    noteIdMap,
    raterIdMap,
    noteRatingIds,
    useGlobalIntercept,
    numFactors,
    logging,
    noteInit,
    userInit,
    globalInterceptInit,
  )

  if specificNoteId is not None:
    # Only used for quick, approximate analysis to score a particular note
    mf_model.freeze_rater_and_global_parameters()
    noteRatingIdsForSpecificNote = noteRatingIds.loc[noteRatingIds[c.noteIdKey] == specificNoteId]
    rating = torch.FloatTensor(noteRatingIdsForSpecificNote[c.helpfulNumKey].values).to(
      mf_model.device
    )
    row = torch.LongTensor(noteRatingIdsForSpecificNote[c.raterIndexKey].values).to(mf_model.device)
    col = torch.LongTensor(noteRatingIdsForSpecificNote[c.noteIndexKey].values).to(mf_model.device)
  else:
    # Normal case
    rating = torch.FloatTensor(noteRatingIds[c.helpfulNumKey].values).to(mf_model.device)
    row = torch.LongTensor(noteRatingIds[c.raterIndexKey].values).to(mf_model.device)
    col = torch.LongTensor(noteRatingIds[c.noteIndexKey].values).to(mf_model.device)

  fit_model(
    mf_model,
    optimizer,
    criterion,
    l2_lambda,
    l2_intercept_multiplier,
    row,
    col,
    rating,
    logging,
  )

  assert mf_model.item_factors.weight.data.cpu().numpy().shape[0] == noteIdMap.shape[0]

  globalIntercept = None
  if useGlobalIntercept:
    globalIntercept = mf_model.global_intercept

  fitNoteParams, fitRaterParams = get_parameters_from_trained_model(
    mf_model, noteIdMap, raterIdMap, flipFactorsForIdentification, numFactors
  )

  return fitNoteParams, fitRaterParams, globalIntercept


def flip_factors_for_identification(
  noteIdMap: pd.DataFrame, raterIdMap: pd.DataFrame, numFactors: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Flip factors if needed, so that the larger group of raters gets a negative factor

  Args:
      noteIdMap (pd.DataFrame)
      raterIdMap (pd.DataFrame)

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: noteIdMap, raterIdMap
  """
  for i in range(1, numFactors + 1):
    noteFactorName = c.note_factor_key(i)
    raterFactorName = c.rater_factor_key(i)

    raterFactors = raterIdMap.loc[~pd.isna(raterIdMap[raterFactorName]), raterFactorName]
    propNegativeRaterFactors = (raterFactors < 0).sum() / (raterFactors != 0).sum()

    if propNegativeRaterFactors < 0.5:
      # Flip all factors, on notes and raters
      noteIdMap[noteFactorName] = noteIdMap[noteFactorName] * -1
      raterIdMap[raterFactorName] = raterIdMap[raterFactorName] * -1

    raterFactors = raterIdMap.loc[~pd.isna(raterIdMap[raterFactorName]), raterFactorName]
    propNegativeRaterFactors = (raterFactors < 0).sum() / (raterFactors != 0).sum()
    assert propNegativeRaterFactors >= 0.5

  return noteIdMap, raterIdMap


"""
TODO: refactor all functions below this one, that have to do with estimating note parameter
uncertainty by adding pseudo-raters, into their own class to improve readability.
There are many variables passed around to every function that it'd be good to store as attributes.
"""


def check_rater_parameters_same(
  mf_model_fixed_raters, raterParams, noteIdMap, raterIdMapWithExtreme, numFactors
):
  noteParamsFromNewModel, raterParamsFromNewModel = get_parameters_from_trained_model(
    mf_model_fixed_raters, noteIdMap, raterIdMapWithExtreme, True, numFactors
  )
  rpCols = [c.raterParticipantIdKey, c.raterInterceptKey, c.raterFactor1Key]
  assert ((raterParamsFromNewModel[rpCols]) == (raterParams[rpCols])).all().all()


def check_note_parameters_same(
  mf_model_fixed_raters, noteParams, noteIdMap, raterIdMap, numFactors
):
  noteParamsFromNewModel, raterParamsFromNewModel = get_parameters_from_trained_model(
    mf_model_fixed_raters, noteIdMap, raterIdMap, True, numFactors
  )
  assert (noteParamsFromNewModel == noteParams).all().all()


def make_extreme_raters(raterParams, raterIdMap):
  raterInterceptValues = [
    raterParams[c.raterInterceptKey].min(),
    0.0,
    raterParams[c.raterInterceptKey].median(),
    raterParams[c.raterInterceptKey].max(),
  ]
  raterFactorValues = [
    raterParams[c.raterFactor1Key].min(),
    raterParams[c.raterFactor1Key].median(),
    0.0,
    raterParams[c.raterFactor1Key].max(),
  ]

  extremeRaters = []
  i = 0
  for raterIntercept in raterInterceptValues:
    for raterFactor in raterFactorValues:
      # These pseudo-raters need to have IDs that don't conflict with real raterParticipantIds
      raterParticipantId = -1 - i
      raterIndex = raterIdMap[c.raterIndexKey].max() + 1 + i

      extremeRaters.append(
        {
          c.raterParticipantIdKey: raterParticipantId,
          c.raterIndexKey: raterIndex,
          c.raterInterceptKey: raterIntercept,
          c.raterFactor1Key: raterFactor,
        }
      )
      i += 1
  return extremeRaters


def add_extreme_raters(raterParams, raterIdMap, extremeRaters):
  # TODO: should concat all at once for efficiency instead of iterative appends
  for i, raterDict in enumerate(extremeRaters):
    if not (raterIdMap[c.raterParticipantIdKey] == raterDict[c.raterParticipantIdKey]).any():
      raterIdMap = pd.concat(
        [
          raterIdMap,
          pd.DataFrame(
            {
              c.raterParticipantIdKey: [raterDict[c.raterParticipantIdKey]],
              c.raterIndexKey: [raterDict[c.raterIndexKey]],
            }
          ),
        ]
      )
    if not (raterParams[c.raterParticipantIdKey] == raterDict[c.raterParticipantIdKey]).any():
      raterParams = pd.concat(
        [
          raterParams,
          pd.DataFrame(
            {
              c.raterParticipantIdKey: [raterDict[c.raterParticipantIdKey]],
              c.raterInterceptKey: [raterDict[c.raterInterceptKey]],
              c.raterFactor1Key: [raterDict[c.raterFactor1Key]],
            }
          ),
        ]
      )
  return raterParams, raterIdMap


def create_new_model_with_extreme_raters_from_original_params(
  ratings,
  noteParams,
  raterParams,
  globalInterceptInit,
  extremeRaters,
  logging=True,
  useGlobalIntercept=True,
  numFactors=1,
  l2_lambda=c.l2_lambda,
  l2_intercept_multiplier=c.l2_intercept_multiplier,
):

  noteIdMap, raterIdMap, noteRatingIds = get_note_and_rater_id_maps(ratings)
  raterParamsWithExtreme, raterIdMapWithExtreme = add_extreme_raters(
    raterParams, raterIdMap, extremeRaters
  )

  noteInit = noteParams.copy(deep=True)
  userInit = raterParamsWithExtreme.copy(deep=True)

  n_users = len(userInit)
  n_items = len(noteInit)
  if logging:
    print("------------------")
    print(f"Users: {n_users}, Notes: {n_items}")

  mf_model_fixed_raters_new = BiasedMatrixFactorization(
    n_users, n_items, use_global_intercept=useGlobalIntercept, n_factors=numFactors
  )

  print(mf_model_fixed_raters_new.device)
  mf_model_fixed_raters_new.to(mf_model_fixed_raters_new.device)

  initialize_parameters(
    mf_model_fixed_raters_new,
    noteIdMap,
    raterIdMapWithExtreme,
    noteInit,
    userInit,
    globalInterceptInit,
    numFactors,
  )
  mf_model_fixed_raters_new.freeze_rater_and_global_parameters()

  notesOnlyOptim = torch.optim.Adam(
    list(mf_model_fixed_raters_new.item_intercepts.parameters())
    + list(mf_model_fixed_raters_new.item_factors.parameters()),
    lr=c.initLearningRate,
  )
  criterion = torch.nn.MSELoss()

  check_note_parameters_same(
    mf_model_fixed_raters_new, noteParams, noteIdMap, raterIdMapWithExtreme, numFactors
  )
  check_rater_parameters_same(
    mf_model_fixed_raters_new, raterParamsWithExtreme, noteIdMap, raterIdMapWithExtreme, numFactors
  )

  return (
    mf_model_fixed_raters_new,
    noteIdMap,
    raterIdMapWithExtreme,
    notesOnlyOptim,
    criterion,
    raterParamsWithExtreme,
  )


def fit_all_notes_with_raters_constant(
  noteRatingIdsWithExtremeRatings,
  noteParams,
  raterParams,
  ratings,
  globalInterceptInit,
  extremeRaters,
  logging=True,
):
  (
    mf_model_fixed_raters,
    noteIdMap,
    raterIdMap,
    notesOnlyOptim,
    criterion,
    raterParamsNew,
  ) = create_new_model_with_extreme_raters_from_original_params(
    ratings,
    noteParams,
    raterParams,
    globalInterceptInit,
    extremeRaters,
    logging=logging,
    useGlobalIntercept=True,
    numFactors=1,
    l2_lambda=c.l2_lambda,
    l2_intercept_multiplier=c.l2_intercept_multiplier,
  )

  rating = torch.FloatTensor(noteRatingIdsWithExtremeRatings[c.helpfulNumKey].values).to(
    mf_model_fixed_raters.device
  )
  row = torch.LongTensor(noteRatingIdsWithExtremeRatings[c.raterIndexKey].values).to(
    mf_model_fixed_raters.device
  )
  col = torch.LongTensor(noteRatingIdsWithExtremeRatings[c.noteIndexKey].values).to(
    mf_model_fixed_raters.device
  )

  fit_model(
    mf_model_fixed_raters,
    notesOnlyOptim,
    criterion,
    c.l2_lambda,
    c.l2_intercept_multiplier,
    row,
    col,
    rating,
    logging=logging,
  )

  # Double check that we kept rater parameters fixed during re-training of note parameters.
  check_rater_parameters_same(
    mf_model_fixed_raters, raterParamsNew, noteIdMap, raterIdMap, c.numFactors
  )

  fitNoteParams, fitRaterParams = get_parameters_from_trained_model(
    mf_model_fixed_raters, noteIdMap, raterIdMap
  )

  return fitNoteParams


def fit_note_params_for_each_dataset_with_extreme_ratings(
  extremeRaters,
  noteRatingIds,
  ratings,
  noteParams,
  raterParams,
  globalInterceptInit,
  logging=True,
  useGlobalIntercept=True,
  numFactors=1,
  l2_lambda=c.l2_lambda,
  l2_intercept_multiplier=c.l2_intercept_multiplier,
  joinOrig=False,
):
  extremeRatingsToAddWithoutNotes = []
  extremeRatingsToAddWithoutNotes.append(
    {
      c.raterInterceptKey: None,
      c.raterFactor1Key: None,
      c.helpfulNumKey: None,
    }
  )
  for r in extremeRaters:
    r[c.raterParticipantIdKey] = str(r[c.raterParticipantIdKey])

    for helpfulNum in (0.0, 1.0):
      r[c.helpfulNumKey] = helpfulNum
      extremeRatingsToAddWithoutNotes.append(r.copy())

  noteParamsList = []
  for ratingToAddWithoutNoteId in extremeRatingsToAddWithoutNotes:
    ## for each rating (ided by raterParticipantId and raterIndex)
    if ratingToAddWithoutNoteId[c.helpfulNumKey] is not None:
      ratingsWithNoteIds = []
      for i, noteRow in noteRatingIds[[c.noteIdKey, c.noteIndexKey]].drop_duplicates().iterrows():
        ratingToAdd = ratingToAddWithoutNoteId.copy()
        ratingToAdd[c.noteIdKey] = noteRow[c.noteIdKey]
        ratingToAdd[c.noteIndexKey] = noteRow[c.noteIndexKey]
        ratingsWithNoteIds.append(ratingToAdd)
      extremeRatingsToAdd = pd.DataFrame(ratingsWithNoteIds).drop(
        [c.raterInterceptKey, c.raterFactor1Key], axis=1
      )
      noteRatingIdsWithExtremeRatings = pd.concat([noteRatingIds, extremeRatingsToAdd])
    else:
      noteRatingIdsWithExtremeRatings = noteRatingIds

    if logging:
      print("------------------")
      print(f"Re-scoring all notes with extra rating added: {ratingToAddWithoutNoteId}")
    fitNoteParams = fit_all_notes_with_raters_constant(
      noteRatingIdsWithExtremeRatings,
      noteParams,
      raterParams,
      ratings,
      globalInterceptInit,
      extremeRaters,
      logging,
    )
    fitNoteParams[c.extraRaterInterceptKey] = ratingToAddWithoutNoteId[c.raterInterceptKey]
    fitNoteParams[c.extraRaterFactor1Key] = ratingToAddWithoutNoteId[c.raterFactor1Key]
    fitNoteParams[c.extraRatingHelpfulNumKey] = ratingToAddWithoutNoteId[c.helpfulNumKey]
    noteParamsList.append(fitNoteParams)

  unp = pd.concat(noteParamsList)
  unp.drop(c.noteIndexKey, axis=1, inplace=True)
  unp = unp.sort_values(by=[c.noteIdKey, c.extraRaterInterceptKey])

  unpAgg = (
    unp[["noteId", "noteIntercept", "noteFactor1"]].groupby("noteId").agg({"min", "median", "max"})
  )

  refitSameRatings = unp[pd.isna(unp[c.extraRaterInterceptKey])][
    [c.noteIdKey, c.noteInterceptKey, c.noteFactor1Key]
  ].set_index(c.noteIdKey)
  refitSameRatings.columns = pd.MultiIndex.from_product([refitSameRatings.columns, ["refit_orig"]])
  n = refitSameRatings.join(unpAgg)

  if joinOrig:
    orig = noteParams[[c.noteIdKey, c.noteInterceptKey, c.noteFactor1Key]].set_index(c.noteIdKey)
    orig.columns = pd.MultiIndex.from_product([orig.columns, ["original"]])
    n = n.join(orig)

  raterFacs = noteRatingIds.merge(raterParams, on=c.raterParticipantIdKey)
  raterFacs["all"] = 1
  raterFacs["neg_fac"] = raterFacs[c.raterFactor1Key] < 0
  raterFacs["pos_fac"] = raterFacs[c.raterFactor1Key] > 0
  r = raterFacs.groupby(c.noteIdKey).sum()[["all", "neg_fac", "pos_fac"]]
  r.columns = pd.MultiIndex.from_product([[c.ratingCountKey], r.columns])
  n = n.join(r)

  def flatten_column_names(c):
    if type(c) == tuple:
      return f"{c[0]}_{c[1]}"
    else:
      return c

  n.columns = [flatten_column_names(c) for c in n.columns]
  n = n[n.columns.sort_values()]

  return unp, n
