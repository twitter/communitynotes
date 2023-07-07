from typing import List, Optional, Tuple

from . import constants as c

import numpy as np
import pandas as pd
import torch


# String Constants.
_noteIndexKey = "noteIndex"
_raterIndexKey = "raterIndex"
_extraRaterInterceptKey = "extraRaterIntercept"
_extraRaterFactor1Key = "extraRaterFactor1"
_extraRatingHelpfulNumKey = "extraRatingHelpfulNum"


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


class MatrixFactorization:
  def __init__(
    self,
    l2_lambda=0.03,
    l2_intercept_multiplier=5,
    initLearningRate=0.2,
    noInitLearningRate=1.0,
    convergence=1e-7,
    numFactors=1,
    useGlobalIntercept=True,
    logging=True,
    flipFactorsForIdentification=True,
  ) -> None:
    """Configure matrix factorization note ranking."""
    self._l2_lambda = l2_lambda
    self._l2_intercept_multiplier = l2_intercept_multiplier
    self._initLearningRate = initLearningRate
    self._noInitLearningRate = noInitLearningRate
    self._convergence = convergence
    self._numFactors = numFactors
    self._useGlobalIntercept = useGlobalIntercept
    self._logging = logging
    self._flipFactorsForIdentification = flipFactorsForIdentification
    self.train_errors: List[float] = []
    self.test_errors: List[float] = []

  def get_note_and_rater_id_maps(
    self,
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
      .rename(columns={0: c.raterParticipantIdKey, "index": _raterIndexKey})
    )

    noteIdMap = (
      pd.DataFrame(noteData[c.noteIdKey].unique())
      .reset_index()
      .set_index(0)
      .reset_index()
      .rename(columns={0: c.noteIdKey, "index": _noteIndexKey})
    )

    noteRatingIds = noteData.merge(noteIdMap, on=c.noteIdKey)
    noteRatingIds = noteRatingIds.merge(raterIdMap, on=c.raterParticipantIdKey)

    return noteIdMap, raterIdMap, noteRatingIds

  def _initialize_parameters(
    self,
    mf_model: BiasedMatrixFactorization,
    noteIdMap: pd.DataFrame,
    raterIdMap: pd.DataFrame,
    noteInit: Optional[pd.DataFrame] = None,
    userInit: Optional[pd.DataFrame] = None,
    globalInterceptInit: Optional[float] = None,
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
        np.expand_dims(noteInit[c.internalNoteInterceptKey].astype(np.float32).values, axis=1)
      )
      mf_model.item_factors.weight.data = torch.tensor(
        noteInit[[c.note_factor_key(i) for i in range(1, self._numFactors + 1)]]
        .astype(np.float32)
        .values
      )

    if userInit is not None:
      print("initializing users")
      userInit = raterIdMap.merge(userInit, on=c.raterParticipantIdKey, how="left")
      mf_model.user_intercepts.weight.data = torch.tensor(
        np.expand_dims(userInit[c.internalRaterInterceptKey].astype(np.float32).values, axis=1)
      )
      mf_model.user_factors.weight.data = torch.tensor(
        userInit[[c.rater_factor_key(i) for i in range(1, self._numFactors + 1)]]
        .astype(np.float32)
        .values
      )

    if globalInterceptInit is not None:
      print("initialized global intercept")
      mf_model.global_intercept = torch.nn.parameter.Parameter(
        torch.ones(1, 1) * globalInterceptInit
      )

  def _get_parameters_from_trained_model(
    self,
    mf_model: BiasedMatrixFactorization,
    noteIdMap: pd.DataFrame,
    raterIdMap: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        mf_model (BiasedMatrixFactorization)
        noteIdMap (pd.DataFrame)
        raterIdMap (pd.DataFrame)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: noteIdMap, raterIdMap
    """
    noteParams = noteIdMap.copy(deep=True)
    raterParams = raterIdMap.copy(deep=True)

    noteParams[c.internalNoteInterceptKey] = mf_model.item_intercepts.weight.data.cpu().numpy()
    raterParams[c.internalRaterInterceptKey] = mf_model.user_intercepts.weight.data.cpu().numpy()

    for i in range(self._numFactors):
      noteParams[c.note_factor_key(i + 1)] = mf_model.item_factors.weight.data.cpu().numpy()[:, i]
      raterParams[c.rater_factor_key(i + 1)] = mf_model.user_factors.weight.data.cpu().numpy()[:, i]

    if self._flipFactorsForIdentification:
      noteParams, raterParams = self._flip_factors_for_identification(noteParams, raterParams)

    return noteParams, raterParams

  def _create_mf_model(
    self,
    noteIdMap: pd.DataFrame,
    raterIdMap: pd.DataFrame,
    noteRatingIds: pd.DataFrame,
    noteInit: Optional[pd.DataFrame] = None,
    userInit: Optional[pd.DataFrame] = None,
    globalInterceptInit: Optional[float] = None,
  ) -> Tuple[BiasedMatrixFactorization, torch.optim.Optimizer, torch.nn.modules.loss._Loss]:
    """Initialize BiasedMatrixFactorization model, optimizer, and loss.

    Args:
        noteIdMap (pd.DataFrame)
        raterIdMap (pd.DataFrame)
        noteRatingIds (pd.DataFrame)
        noteInit (pd.DataFrame, optional)
        userInit (pd.DataFrame, optional)
        globalIntercept (float, optional)

    Returns:
        Tuple[BiasedMatrixFactorization, torch.optim.Optimizer, torch.nn.modules.loss._Loss]
    """
    n_users = noteRatingIds[_raterIndexKey].nunique()
    n_items = noteRatingIds[_noteIndexKey].nunique()
    if self._logging:
      print("------------------")
      print(f"Users: {n_users}, Notes: {n_items}")

    criterion = torch.nn.MSELoss()

    mf_model = BiasedMatrixFactorization(
      n_users, n_items, use_global_intercept=self._useGlobalIntercept, n_factors=self._numFactors
    )

    self._initialize_parameters(
      mf_model, noteIdMap, raterIdMap, noteInit, userInit, globalInterceptInit
    )

    if (noteInit is not None) and (userInit is not None):
      optimizer = torch.optim.Adam(
        mf_model.parameters(), lr=self._initLearningRate
      )  # smaller learning rate
    else:
      optimizer = torch.optim.Adam(
        mf_model.parameters(), lr=self._noInitLearningRate
      )  # learning rate

    print(mf_model.device)
    mf_model.to(mf_model.device)

    return mf_model, optimizer, criterion

  def _get_train_validate_sets(
    self,
    row: torch.LongTensor,
    col: torch.LongTensor,
    rating: torch.FloatTensor,
    validate_percent: Optional[float] = None,
  ) -> Tuple[
    torch.LongTensor,
    torch.LongTensor,
    torch.FloatTensor,
    torch.LongTensor,
    torch.LongTensor,
    torch.FloatTensor,
  ]:
    if validate_percent is not None:
      random_indices = np.random.permutation(np.arange(len(row)))
      validate_indices = random_indices[: int(validate_percent * len(row))]
      train_indices = random_indices[int(validate_percent * len(row)) :]
      row_validate = row[validate_indices]
      col_validate = col[validate_indices]
      rating_validate = rating[validate_indices]
      row_train = row[train_indices]
      col_train = col[train_indices]
      rating_train = rating[train_indices]
    else:
      row_train = row
      col_train = col
      rating_train = rating
      row_validate = None
      col_validate = None
      rating_validate = None
    return row_train, col_train, rating_train, row_validate, col_validate, rating_validate

  def _compute_and_print_loss(
    self,
    mf_model: BiasedMatrixFactorization,
    loss_value: float,
    criterion: torch.nn.modules.loss._Loss,
    row_train: torch.LongTensor,
    col_train: torch.LongTensor,
    rating_train: torch.FloatTensor,
    row_validate: Optional[torch.LongTensor],
    col_validate: Optional[torch.LongTensor],
    rating_validate: Optional[torch.FloatTensor],
    epoch: int,
    final: bool = False,
  ) -> Tuple[float, float, Optional[float]]:
    y_pred = mf_model(row_train, col_train)
    train_loss_value = criterion(y_pred, rating_train).item()
    if row_validate is not None:
      y_pred_validate = mf_model(row_validate, col_validate)
      validate_loss_value = criterion(y_pred_validate, rating_validate).item()
    else:
      validate_loss_value = None

    if self._logging:
      print("epoch", epoch, loss_value)
      print("TRAIN FIT LOSS: ", train_loss_value)
      if validate_loss_value is not None:
        print("VALIDATE FIT LOSS: ", validate_loss_value)

      if final == True:
        self.test_errors.append(loss_value)
        self.train_errors.append(train_loss_value)
    return train_loss_value, loss_value, validate_loss_value

  def _fit_model(
    self,
    mf_model: BiasedMatrixFactorization,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    row: torch.LongTensor,
    col: torch.LongTensor,
    rating: torch.FloatTensor,
    validate_percent: Optional[float] = None,
  ) -> Tuple[float, float, Optional[float]]:
    """Run gradient descent to train the model.

    Args:
        mf_model (BiasedMatrixFactorization)
        optimizer (torch.optim.Optimizer)
        criterion (torch.nn.modules.loss._Loss)
        row (torch.LongTensor)
        col (torch.LongTensor)
        rating (torch.FloatTensor)
    """
    (
      row_train,
      col_train,
      rating_train,
      row_validate,
      col_validate,
      rating_validate,
    ) = self._get_train_validate_sets(row, col, rating, validate_percent)

    l2_lambda_intercept = self._l2_lambda * self._l2_intercept_multiplier
    prev_loss = 1e10

    y_pred = mf_model(row_train, col_train)
    loss = criterion(y_pred, rating_train)
    l2_reg_loss = torch.tensor(0.0).to(mf_model.device)

    for name, param in mf_model.named_parameters():
      if "intercept" in name:
        l2_reg_loss += l2_lambda_intercept * (param**2).mean()
      else:
        l2_reg_loss += self._l2_lambda * (param**2).mean()

    loss += l2_reg_loss

    epoch = 0

    while (abs(loss.item() - prev_loss) > self._convergence) and (
      not (epoch > 100 and loss.item() > prev_loss)
    ):

      prev_loss = loss.item()

      # Backpropagate
      loss.backward()

      # Update the parameters
      optimizer.step()

      # Set gradients to zero
      optimizer.zero_grad()

      # Predict and calculate loss
      y_pred = mf_model(row_train, col_train)
      loss = criterion(y_pred, rating_train)
      l2_reg_loss = torch.tensor(0.0).to(mf_model.device)

      for name, param in mf_model.named_parameters():
        if "intercept" in name:
          l2_reg_loss += l2_lambda_intercept * (param**2).mean()
        else:
          l2_reg_loss += self._l2_lambda * (param**2).mean()

      loss += l2_reg_loss

      if epoch % 50 == 0:
        self._compute_and_print_loss(
          mf_model,
          loss.item(),
          criterion,
          row_train,
          col_train,
          rating_train,
          row_validate,
          col_validate,
          rating_validate,
          epoch,
          final=False,
        )

      epoch += 1

    if self._logging:
      print("Num epochs:", epoch)
    return self._compute_and_print_loss(
      mf_model,
      loss.item(),
      criterion,
      row_train,
      col_train,
      rating_train,
      row_validate,
      col_validate,
      rating_validate,
      epoch,
      final=True,
    )

  def run_mf(
    self,
    ratings: pd.DataFrame,
    noteInit: pd.DataFrame = None,
    userInit: pd.DataFrame = None,
    globalInterceptInit: Optional[float] = None,
    specificNoteId: Optional[int] = None,
    validatePercent: Optional[float] = None,
  ):
    """Train matrix factorization model.

    See https://twitter.github.io/communitynotes/ranking-notes/#matrix-factorization

    Args:
        ratings (pd.DataFrame): pre-filtered ratings to train on
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
    noteIdMap, raterIdMap, noteRatingIds = self.get_note_and_rater_id_maps(ratings)

    mf_model, optimizer, criterion = self._create_mf_model(
      noteIdMap,
      raterIdMap,
      noteRatingIds,
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
      row = torch.LongTensor(noteRatingIdsForSpecificNote[_raterIndexKey].values).to(
        mf_model.device
      )
      col = torch.LongTensor(noteRatingIdsForSpecificNote[_noteIndexKey].values).to(mf_model.device)
    else:
      # Normal case
      rating = torch.FloatTensor(noteRatingIds[c.helpfulNumKey].values).to(mf_model.device)
      row = torch.LongTensor(noteRatingIds[_raterIndexKey].values).to(mf_model.device)
      col = torch.LongTensor(noteRatingIds[_noteIndexKey].values).to(mf_model.device)

    train_loss, loss, validate_loss = self._fit_model(
      mf_model, optimizer, criterion, row, col, rating, validatePercent
    )

    assert mf_model.item_factors.weight.data.cpu().numpy().shape[0] == noteIdMap.shape[0]

    globalIntercept = None
    if self._useGlobalIntercept:
      globalIntercept = mf_model.global_intercept
      if self._logging:
        print("Global Intercept: ", globalIntercept.item())

    fitNoteParams, fitRaterParams = self._get_parameters_from_trained_model(
      mf_model, noteIdMap, raterIdMap
    )

    fitRaterParams.drop(_raterIndexKey, axis=1, inplace=True)
    if validatePercent is None:
      return fitNoteParams, fitRaterParams, globalIntercept
    else:
      return fitNoteParams, fitRaterParams, globalIntercept, train_loss, loss, validate_loss

  def _flip_factors_for_identification(
    self, noteIdMap: pd.DataFrame, raterIdMap: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flip factors if needed, so that the larger group of raters gets a negative factor

    Args:
        noteIdMap (pd.DataFrame)
        raterIdMap (pd.DataFrame)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: noteIdMap, raterIdMap
    """
    for i in range(1, self._numFactors + 1):
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

  def _check_rater_parameters_same(
    self, mf_model_fixed_raters, raterParams, noteIdMap, raterIdMapWithExtreme
  ):
    noteParamsFromNewModel, raterParamsFromNewModel = self._get_parameters_from_trained_model(
      mf_model_fixed_raters, noteIdMap, raterIdMapWithExtreme
    )
    rpCols = [c.raterParticipantIdKey, c.internalRaterInterceptKey, c.internalRaterFactor1Key]
    assert ((raterParamsFromNewModel[rpCols]) == (raterParams[rpCols])).all().all()

  def _check_note_parameters_same(self, mf_model_fixed_raters, noteParams, noteIdMap, raterIdMap):
    noteParamsFromNewModel, raterParamsFromNewModel = self._get_parameters_from_trained_model(
      mf_model_fixed_raters, noteIdMap, raterIdMap
    )
    assert (noteParamsFromNewModel == noteParams).all().all()

  def make_extreme_raters(self, raterParams, raterIdMap):
    raterInterceptValues = [
      raterParams[c.internalRaterInterceptKey].min(),
      raterParams[c.internalRaterInterceptKey].max(),
    ]
    raterFactorValues = [
      raterParams[c.internalRaterFactor1Key].min(),
      0.0,
      raterParams[c.internalRaterFactor1Key].max(),
    ]

    extremeRaters = []
    i = 0
    for raterIntercept in raterInterceptValues:
      for raterFactor in raterFactorValues:
        # These pseudo-raters need to have IDs that don't conflict with real raterParticipantIds
        raterParticipantId = -1 - i
        raterIndex = raterIdMap[_raterIndexKey].max() + 1 + i

        extremeRaters.append(
          {
            c.raterParticipantIdKey: raterParticipantId,
            _raterIndexKey: raterIndex,
            c.internalRaterInterceptKey: raterIntercept,
            c.internalRaterFactor1Key: raterFactor,
          }
        )
        i += 1
    return extremeRaters

  def _add_extreme_raters(self, raterParams, raterIdMap, extremeRaters):
    # TODO: should concat all at once for efficiency instead of iterative appends
    for i, raterDict in enumerate(extremeRaters):
      if not (raterIdMap[c.raterParticipantIdKey] == raterDict[c.raterParticipantIdKey]).any():
        raterIdMap = pd.concat(
          [
            raterIdMap,
            pd.DataFrame(
              {
                c.raterParticipantIdKey: [raterDict[c.raterParticipantIdKey]],
                _raterIndexKey: [raterDict[_raterIndexKey]],
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
                c.internalRaterInterceptKey: [raterDict[c.internalRaterInterceptKey]],
                c.internalRaterFactor1Key: [raterDict[c.internalRaterFactor1Key]],
              }
            ),
          ]
        )
    return raterParams, raterIdMap

  def _create_new_model_with_extreme_raters_from_original_params(
    self,
    ratings,
    noteParams,
    raterParams,
    globalInterceptInit,
    extremeRaters,
  ):

    noteIdMap, raterIdMap, noteRatingIds = self.get_note_and_rater_id_maps(ratings)
    raterParamsWithExtreme, raterIdMapWithExtreme = self._add_extreme_raters(
      raterParams, raterIdMap, extremeRaters
    )

    noteInit = noteParams.copy(deep=True)
    userInit = raterParamsWithExtreme.copy(deep=True)

    n_users = len(userInit)
    n_items = len(noteInit)
    if self._logging:
      print("------------------")
      print(f"Users: {n_users}, Notes: {n_items}")

    mf_model_fixed_raters_new = BiasedMatrixFactorization(
      n_users, n_items, use_global_intercept=self._useGlobalIntercept, n_factors=self._numFactors
    )

    print(mf_model_fixed_raters_new.device)
    mf_model_fixed_raters_new.to(mf_model_fixed_raters_new.device)

    self._initialize_parameters(
      mf_model_fixed_raters_new,
      noteIdMap,
      raterIdMapWithExtreme,
      noteInit,
      userInit,
      globalInterceptInit,
    )
    mf_model_fixed_raters_new.freeze_rater_and_global_parameters()

    notesOnlyOptim = torch.optim.Adam(
      list(mf_model_fixed_raters_new.item_intercepts.parameters())
      + list(mf_model_fixed_raters_new.item_factors.parameters()),
      lr=self._initLearningRate,
    )
    criterion = torch.nn.MSELoss()

    self._check_note_parameters_same(
      mf_model_fixed_raters_new, noteParams, noteIdMap, raterIdMapWithExtreme
    )
    self._check_rater_parameters_same(
      mf_model_fixed_raters_new, raterParamsWithExtreme, noteIdMap, raterIdMapWithExtreme
    )

    return (
      mf_model_fixed_raters_new,
      noteIdMap,
      raterIdMapWithExtreme,
      notesOnlyOptim,
      criterion,
      raterParamsWithExtreme,
    )

  def _fit_all_notes_with_raters_constant(
    self,
    noteRatingIdsWithExtremeRatings,
    noteParams,
    raterParams,
    ratings,
    globalInterceptInit,
    extremeRaters,
  ):
    (
      mf_model_fixed_raters,
      noteIdMap,
      raterIdMap,
      notesOnlyOptim,
      criterion,
      raterParamsNew,
    ) = self._create_new_model_with_extreme_raters_from_original_params(
      ratings,
      noteParams,
      raterParams,
      globalInterceptInit,
      extremeRaters,
    )

    rating = torch.FloatTensor(noteRatingIdsWithExtremeRatings[c.helpfulNumKey].values).to(
      mf_model_fixed_raters.device
    )
    row = torch.LongTensor(noteRatingIdsWithExtremeRatings[_raterIndexKey].values).to(
      mf_model_fixed_raters.device
    )
    col = torch.LongTensor(noteRatingIdsWithExtremeRatings[_noteIndexKey].values).to(
      mf_model_fixed_raters.device
    )

    self._fit_model(
      mf_model_fixed_raters,
      notesOnlyOptim,
      criterion,
      row,
      col,
      rating,
    )

    # Double check that we kept rater parameters fixed during re-training of note parameters.
    self._check_rater_parameters_same(mf_model_fixed_raters, raterParamsNew, noteIdMap, raterIdMap)

    fitNoteParams, fitRaterParams = self._get_parameters_from_trained_model(
      mf_model_fixed_raters, noteIdMap, raterIdMap
    )

    return fitNoteParams

  def fit_note_params_for_each_dataset_with_extreme_ratings(
    self,
    extremeRaters,
    noteRatingIds,
    ratings,
    noteParams,
    raterParams,
    globalInterceptInit,
    joinOrig=False,
  ):
    extremeRatingsToAddWithoutNotes = []
    extremeRatingsToAddWithoutNotes.append(
      {
        c.internalRaterInterceptKey: None,
        c.internalRaterFactor1Key: None,
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
        for i, noteRow in noteRatingIds[[c.noteIdKey, _noteIndexKey]].drop_duplicates().iterrows():
          ratingToAdd = ratingToAddWithoutNoteId.copy()
          ratingToAdd[c.noteIdKey] = noteRow[c.noteIdKey]
          ratingToAdd[_noteIndexKey] = noteRow[_noteIndexKey]
          ratingsWithNoteIds.append(ratingToAdd)
        extremeRatingsToAdd = pd.DataFrame(ratingsWithNoteIds).drop(
          [c.internalRaterInterceptKey, c.internalRaterFactor1Key], axis=1
        )
        noteRatingIdsWithExtremeRatings = pd.concat([noteRatingIds, extremeRatingsToAdd])
      else:
        noteRatingIdsWithExtremeRatings = noteRatingIds

      if self._logging:
        print("------------------")
        print(f"Re-scoring all notes with extra rating added: {ratingToAddWithoutNoteId}")
      fitNoteParams = self._fit_all_notes_with_raters_constant(
        noteRatingIdsWithExtremeRatings,
        noteParams,
        raterParams,
        ratings,
        globalInterceptInit,
        extremeRaters,
      )
      fitNoteParams[_extraRaterInterceptKey] = ratingToAddWithoutNoteId[c.internalRaterInterceptKey]
      fitNoteParams[_extraRaterFactor1Key] = ratingToAddWithoutNoteId[c.internalRaterFactor1Key]
      fitNoteParams[_extraRatingHelpfulNumKey] = ratingToAddWithoutNoteId[c.helpfulNumKey]
      noteParamsList.append(fitNoteParams)

    unp = pd.concat(noteParamsList)
    unp.drop(_noteIndexKey, axis=1, inplace=True)
    unp = unp.sort_values(by=[c.noteIdKey, _extraRaterInterceptKey])

    unpAgg = (
      unp[[c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]]
      .groupby(c.noteIdKey)
      .agg({"min", "median", "max"})
    )

    refitSameRatings = unp[pd.isna(unp[_extraRaterInterceptKey])][
      [c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]
    ].set_index(c.noteIdKey)
    refitSameRatings.columns = pd.MultiIndex.from_product(
      [refitSameRatings.columns, ["refit_orig"]]
    )
    n = refitSameRatings.join(unpAgg)

    if joinOrig:
      orig = noteParams[
        [c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]
      ].set_index(c.noteIdKey)
      orig.columns = pd.MultiIndex.from_product([orig.columns, ["original"]])
      n = n.join(orig)

    raterFacs = noteRatingIds.merge(raterParams, on=c.raterParticipantIdKey)
    raterFacs["all"] = 1
    raterFacs["neg_fac"] = raterFacs[c.internalRaterFactor1Key] < 0
    raterFacs["pos_fac"] = raterFacs[c.internalRaterFactor1Key] > 0
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
