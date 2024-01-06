import os
import dataclasses
from typing import List, Optional, Tuple

from .. import constants as c
from .model import BiasedMatrixFactorization, ModelData

import numpy as np
import pandas as pd
import torch

@dataclasses.dataclass
class Constants:
  noteIndexKey = "noteIndex"
  raterIndexKey = "raterIndex"


class MatrixFactorization:
  """
  This class implements a Matrix Factorization model, commonly used in recommendation systems 
  and collaborative filtering. It decomposes a matrix into the product of two lower-dimensional matrices, 
  capturing latent factors in the data.

  Attributes:
      l2_lambda (float): Regularization parameter for L2 regularization.
      l2_intercept_multiplier (float): Multiplier for the intercept in L2 regularization.
      init_lr (float): Initial learning rate for the optimizer.
      noinit_lr (float): Learning rate used when no initial values are provided.
      convergence (float): Convergence threshold for the training process.
      num_factors (int): Number of latent factors to model.
      use_global_intercept (bool): Flag to use a global intercept in the model.
      use_sigmoid_crossentropy (bool): Use sigmoid cross-entropy loss if True, else mean squared error loss.
      logging (bool): Enable or disable logging.
      flip_factor_identification (bool): Adjust factors for model identification.
      model (BiasedMatrixFactorization, optional): An instance of a biased matrix factorization model.
      feature_cols (List[str]): Feature columns to use in the model.
      label_col (str): Label column in the data.
      pos_weight (optional): Positive weight parameter for the loss function.

  Methods:
      get_final_train_error(): Returns the final training error after model fitting.
      get_new_mf_with_same_args(): Creates a new instance of MatrixFactorization with the same configuration.
      _initialize_note_and_rater_id_maps(ratings): Initializes mappings for note and rater IDs based on the provided ratings DataFrame.
      get_note_and_rater_id_maps(ratings): Extracts and returns mappings for note and rater IDs along with processed rating features and labels.
      _initialize_parameters(): Initializes or resets the model parameters with given initial values or defaults.
      _get_parameters_from_trained_model(): Retrieves parameters from the trained model for analysis or further use.
      _create_mf_model(): Initializes the matrix factorization model and its parameters.
      _compute_and_print_loss(): Computes and logs the loss during training, useful for monitoring model performance.
      _create_train_validate_sets(): Splits the data into training and validation sets for model fitting.
      _fit_model(): Executes the model training process, adjusting parameters to minimize the loss.
      prepare_features_and_labels(): Prepares features and labels from the dataset for model training.
      run_mf(): Main method to run matrix factorization on provided data, returning trained model parameters and performance metrics.
      _flip_factors_for_identification(): Adjusts factor sign for model identifiability and interpretation.
  """
     
  def __init__(
    self,
    l2_lambda: float,
    l2_intercept_multiplier: int,
    init_lr: float,
    noinit_lr: float,
    convergence: float,
    num_factors: int,
    use_global_intercept: bool,
    use_sigmoid_crossentropy: bool,
    logging: bool,
    flip_factor_identification: bool,
    model: Optional[BiasedMatrixFactorization] = None,
    feature_cols: List[str] = [c.noteIdKey, c.raterParticipantIdKey],
    label_col: str = c.helpfulNumKey,
    pos_weight: Optional[float] = None,
    ) -> None:
    self._l2_lambda = l2_lambda
    self._l2_intercept_multiplier = l2_intercept_multiplier
    self._init_lr = init_lr
    self._noinit_lr = noinit_lr
    self._convergence = convergence
    self._num_factors = num_factors
    self._use_global_intercept = use_global_intercept
    self._logging = logging
    self._flip_factor_identification = flip_factor_identification
    self._feature_cols = feature_cols
    self._label_col = label_col
    self._use_sigmoid_crossentropy = use_sigmoid_crossentropy
    self._pos_weight = pos_weight

    if self._use_sigmoid_crossentropy:
      if self._pos_weight:
        if logging:
          print(f"Using pos weight: {self._pos_weight} with BCEWithLogitsLoss")
        self.criterion = torch.nn.BCEWithLogitsLoss(
          pos_weight=torch.Tensor(np.array(self._pos_weight))
        )
      else:
        if logging:
          print("Using BCEWithLogitsLoss")
        self.criterion = torch.nn.BCEWithLogitsLoss()
    else:
      if self._pos_weight:
        raise ValueError("pos_weight is not supported for MSELoss")
      self.criterion = torch.nn.MSELoss()

    self.train_errors: List[float] = []
    self.test_errors: List[float] = []
    self.mf_model = model

    self.modelData: Optional[ModelData] = None
    self.trainModelData: Optional[ModelData] = None
    self.validateModelData: Optional[ModelData] = None

  def get_final_train_error(self) -> Optional[float]: return self.train_errors[-1] if self.train_errors else None

  def get_new_mf_with_same_args(self):
    return MatrixFactorization(
      l2_lambda=self._l2_lambda,
      l2_intercept_multiplier=self._l2_intercept_multiplier,
      init_lr=self._init_lr,
      noinit_lr=self._noinit_lr,
      convergence=self._convergence,
      num_factors=self._num_factors,
      use_global_intercept=self._use_global_intercept,
      logging=self._logging,
      flip_factor_identification=self._flip_factor_identification,
      model=None,
      feature_cols=self._feature_cols,
      label_col=self._label_col,
    )

  def _initialize_note_and_rater_id_maps(
    self,
    ratings: pd.DataFrame,
  ) -> None:
    self.noteIdMap, self.raterIdMap, self.ratingFeaturesAndLabels = self.get_note_and_rater_id_maps(
      ratings
    )

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
    noteData = ratings[self._feature_cols + [self._label_col]]
    assert not pd.isna(noteData).values.any(), "noteData must not contain nan values"

    raterIdMap = (
      pd.DataFrame(noteData[c.raterParticipantIdKey].unique())
      .reset_index()
      .set_index(0)
      .reset_index()
      .rename(columns={0: c.raterParticipantIdKey, "index": Constants.raterIndexKey})
    )

    noteIdMap = (
      pd.DataFrame(noteData[c.noteIdKey].unique())
      .reset_index()
      .set_index(0)
      .reset_index()
      .rename(columns={0: c.noteIdKey, "index": Constants.noteIndexKey})
    )

    ratingFeaturesAndLabels = noteData.merge(noteIdMap, on=c.noteIdKey)
    ratingFeaturesAndLabels = ratingFeaturesAndLabels.merge(raterIdMap, on=c.raterParticipantIdKey)

    return noteIdMap, raterIdMap, ratingFeaturesAndLabels

  def _initialize_parameters(
    self,
    noteInit: Optional[pd.DataFrame] = None,
    userInit: Optional[pd.DataFrame] = None,
    globalInterceptInit: Optional[float] = None,
  ) -> None:
    """Overwrite the parameters of the model with the given initializations.
    Set parameters to 0.0 if they were not set in the passed initializations.

    Args:
        mf_model (BiasedMatrixFactorization)
        noteIdMap (pd.DataFrame)
        raterIdMap (pd.DataFrame)
        noteInit (pd.DataFrame, optional)
        userInit (pd.DataFrame, optional)
        globalInterceptInit (float, optional)
    """
    assert self.mf_model is not None
    if noteInit is not None:
      if self._logging:
        print("initializing notes")
      noteInit = self.noteIdMap.merge(noteInit, on=c.noteIdKey, how="left")

      noteInit[c.internalNoteInterceptKey].fillna(0.0, inplace=True)
      self.mf_model.note_intercepts.weight.data = torch.tensor(
        np.expand_dims(noteInit[c.internalNoteInterceptKey].astype(np.float32).values, axis=1)
      )

      for i in range(1, self._num_factors + 1):
        noteInit[c.note_factor_key(i)].fillna(0.0, inplace=True)
      self.mf_model.note_factors.weight.data = torch.tensor(
        noteInit[[c.note_factor_key(i) for i in range(1, self._num_factors + 1)]]
        .astype(np.float32)
        .values
      )

    if userInit is not None:
      if self._logging:
        print("initializing users")
      userInit = self.raterIdMap.merge(userInit, on=c.raterParticipantIdKey, how="left")

      userInit[c.internalRaterInterceptKey].fillna(0.0, inplace=True)
      self.mf_model.user_intercepts.weight.data = torch.tensor(
        np.expand_dims(userInit[c.internalRaterInterceptKey].astype(np.float32).values, axis=1)
      )

      for i in range(1, self._num_factors + 1):
        userInit[c.rater_factor_key(i)].fillna(0.0, inplace=True)
      self.mf_model.user_factors.weight.data = torch.tensor(
        userInit[[c.rater_factor_key(i) for i in range(1, self._num_factors + 1)]]
        .astype(np.float32)
        .values
      )

    if globalInterceptInit is not None:
      if self._logging:
        print("initialized global intercept")
      self.mf_model.global_intercept = torch.nn.parameter.Parameter(
        torch.ones(1, 1) * globalInterceptInit
      )

  def _get_parameters_from_trained_model(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: noteIdMap, raterIdMap
    """
    assert self.mf_model is not None
    noteParams = self.noteIdMap.copy(deep=True)
    raterParams = self.raterIdMap.copy(deep=True)

    noteParams[c.internalNoteInterceptKey] = self.mf_model.note_intercepts.weight.data.cpu().numpy()
    raterParams[
      c.internalRaterInterceptKey
    ] = self.mf_model.user_intercepts.weight.data.cpu().numpy()

    for i in range(self._num_factors):
      noteParams[c.note_factor_key(i + 1)] = self.mf_model.note_factors.weight.data.cpu().numpy()[
        :, i
      ]
      raterParams[c.rater_factor_key(i + 1)] = self.mf_model.user_factors.weight.data.cpu().numpy()[
        :, i
      ]

    if self._flip_factor_identification:
      noteParams, raterParams = self._flip_factors_for_identification(noteParams, raterParams)

    return noteParams, raterParams

  def _create_mf_model(
    self,
    noteInit: Optional[pd.DataFrame] = None,
    userInit: Optional[pd.DataFrame] = None,
    globalInterceptInit: Optional[float] = None,
  ) -> None:
    """Initialize BiasedMatrixFactorization model and optimizer.

    Args:
        noteIdMap (pd.DataFrame)
        raterIdMap (pd.DataFrame)
        ratingFeaturesAndLabels (pd.DataFrame)
        noteInit (pd.DataFrame, optional)
        userInit (pd.DataFrame, optional)
        globalIntercept (float, optional)
    """
    self._instantiate_biased_mf_model()
    assert self.mf_model is not None
    self._initialize_parameters(noteInit, userInit, globalInterceptInit)

    if (noteInit is not None) and (userInit is not None):
      self.optimizer = torch.optim.Adam(
        self.mf_model.parameters(), lr=self._init_lr
      )  # smaller learning rate
    else:
      self.optimizer = torch.optim.Adam(self.mf_model.parameters(), lr=self._noinit_lr)
    if self._logging:
      print(self.mf_model.device)
    self.mf_model.to(self.mf_model.device)

  def _instantiate_biased_mf_model(self):
    n_users = self.ratingFeaturesAndLabels[Constants.raterIndexKey].nunique()
    n_notes = self.ratingFeaturesAndLabels[Constants.noteIndexKey].nunique()
    self.mf_model = BiasedMatrixFactorization(
      n_users,
      n_notes,
      use_global_intercept=self._use_global_intercept,
      n_factors=self._num_factors,
      logging=self._logging,
    )
    if self._logging:
      print("------------------")
      print(f"Users: {n_users}, Notes: {n_notes}")

  def _compute_and_print_loss(
    self,
    loss_value: float,
    epoch: int,
    final: bool = False,
  ) -> Tuple[float, float, Optional[float]]:
    assert self.mf_model is not None
    assert self.trainModelData is not None
    y_pred = self.mf_model(self.trainModelData)
    train_loss_value = self.criterion(y_pred, self.trainModelData.rating_labels).item()
    if self.validateModelData is not None:
      y_pred_validate = self.mf_model(self.validateModelData)
      validate_loss_value = self.criterion(
        y_pred_validate, self.validateModelData.rating_labels
      ).item()
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

  def _create_train_validate_sets(
    self,
    validate_percent: Optional[float] = None,
  ):
    assert (self.modelData is not None) and (
      self.modelData.user_indexes is not None
    ), "modelData and modelData.user_indexes must be set before calling _create_train_validate_sets"
    if validate_percent is not None:
      random_indices = np.random.permutation(np.arange(len(self.modelData.user_indexes)))
      validate_indices = random_indices[: int(validate_percent * len(self.modelData.user_indexes))]
      train_indices = random_indices[int(validate_percent * len(self.modelData.user_indexes)) :]

      # Replace fields in dataclasses in a general way so new fields can be added without rewriting this code.
      self.trainModelData = dataclasses.replace(self.modelData)
      for field in dataclasses.fields(self.trainModelData):
        currentValue = getattr(self.trainModelData, field.name)
        setattr(self.trainModelData, field.name, currentValue[train_indices])

      self.validateModelData = dataclasses.replace(self.modelData)
      for field in dataclasses.fields(self.validateModelData):
        currentValue = getattr(self.validateModelData, field.name)
        setattr(self.validateModelData, field.name, currentValue[validate_indices])
    else:
      self.trainModelData = self.modelData

  def _fit_model(
    self,
    validate_percent: Optional[float] = None,
    print_interval: int = 20,
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
    assert self.mf_model is not None
    self._create_train_validate_sets()
    assert self.trainModelData is not None

    l2_lambda_intercept = self._l2_lambda * self._l2_intercept_multiplier
    prev_loss = 1e10

    y_pred = self.mf_model(self.trainModelData)
    loss = self.criterion(y_pred, self.trainModelData.rating_labels)
    l2_reg_loss = torch.tensor(0.0).to(self.mf_model.device)

    for name, param in self.mf_model.named_parameters():
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
      self.optimizer.step()

      # Set gradients to zero
      self.optimizer.zero_grad()

      # Predict and calculate loss
      y_pred = self.mf_model(self.trainModelData)
      loss = self.criterion(y_pred, self.trainModelData.rating_labels)
      l2_reg_loss = torch.tensor(0.0).to(self.mf_model.device)

      for name, param in self.mf_model.named_parameters():
        if "intercept" in name:
          l2_reg_loss += l2_lambda_intercept * (param**2).mean()
        else:
          l2_reg_loss += self._l2_lambda * (param**2).mean()

      loss += l2_reg_loss

      if epoch % print_interval == 0:
        self._compute_and_print_loss(loss.item(), epoch, final=False)

      epoch += 1

    if self._logging:
      print("Num epochs:", epoch)
    return self._compute_and_print_loss(loss.item(), epoch, final=True)

  def prepare_features_and_labels(
    self,
    specificNoteId: Optional[int] = None,
  ) -> None:
    assert self.mf_model is not None
    ratingFeaturesAndLabels = self.ratingFeaturesAndLabels
    if specificNoteId is not None:
      ratingFeaturesAndLabels = self.ratingFeaturesAndLabels.loc[
        self.ratingFeaturesAndLabels[c.noteIdKey] == specificNoteId
      ]

    rating_labels = torch.FloatTensor(ratingFeaturesAndLabels[self._label_col].values).to(
      self.mf_model.device
    )
    user_indexes = torch.LongTensor(ratingFeaturesAndLabels[Constants.raterIndexKey].values).to(
      self.mf_model.device
    )
    note_indexes = torch.LongTensor(ratingFeaturesAndLabels[Constants.noteIndexKey].values).to(
      self.mf_model.device
    )
    self.modelData = ModelData(rating_labels, user_indexes, note_indexes)

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
        globalInterceptInit (float, optional).
        specificNoteId (int, optional) Do approximate analysis to score a particular note

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, float]:
          noteParams: contains one row per note, including noteId and learned note parameters
          raterParams: contains one row per rating, including raterId and learned rater parameters
          globalIntercept: learned global intercept parameter
    """
    self._initialize_note_and_rater_id_maps(ratings)

    self._create_mf_model(noteInit, userInit, globalInterceptInit)
    assert self.mf_model is not None

    if specificNoteId is not None:
      self.mf_model.freeze_rater_and_global_parameters()
    self.prepare_features_and_labels(specificNoteId)

    train_loss, loss, validate_loss = self._fit_model(validatePercent)

    assert self.mf_model.note_factors.weight.data.cpu().numpy().shape[0] == self.noteIdMap.shape[0]

    globalIntercept = None
    if self._use_global_intercept:
      globalIntercept = self.mf_model.global_intercept
      if self._logging:
        print("Global Intercept: ", globalIntercept.item())

    fitNoteParams, fitRaterParams = self._get_parameters_from_trained_model()

    fitRaterParams.drop(Constants.raterIndexKey, axis=1, inplace=True)
    if validatePercent is None:
      return fitNoteParams, fitRaterParams, globalIntercept
    else:
      return fitNoteParams, fitRaterParams, globalIntercept, train_loss, loss, validate_loss

  def _flip_factors_for_identification(
    self, noteParams: pd.DataFrame, raterParams: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flip factors if needed, so that the larger group of raters gets a negative factor

    Args:
        noteParams (pd.DataFrame)
        raterParams (pd.DataFrame)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: noteParams, raterParams
    """
    for i in range(1, self._num_factors + 1):
      noteFactorName = c.note_factor_key(i)
      raterFactorName = c.rater_factor_key(i)

      raterFactors = raterParams.loc[~pd.isna(raterParams[raterFactorName]), raterFactorName]
      propNegativeRaterFactors = (raterFactors < 0).sum() / (raterFactors != 0).sum()

      if propNegativeRaterFactors < 0.5:
        # Flip all factors, on notes and raters
        noteParams[noteFactorName] = noteParams[noteFactorName] * -1
        raterParams[raterFactorName] = raterParams[raterFactorName] * -1

      raterFactors = raterParams.loc[~pd.isna(raterParams[raterFactorName]), raterFactorName]
      propNegativeRaterFactors = (raterFactors < 0).sum() / (raterFactors != 0).sum()
      assert propNegativeRaterFactors >= 0.5

    return noteParams, raterParams
