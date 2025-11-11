import dataclasses
import logging
import time
from typing import List, Optional, Tuple

from .. import constants as c
from .model import BiasedMatrixFactorization, ModelData
from .normalized_loss import NormalizedLoss
from .wandb_utils import wandb

import numpy as np
import pandas as pd
import torch


logger = logging.getLogger("birdwatch.matrix_factorization")
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class Constants:
  noteIndexKey = "noteIndex"
  raterIndexKey = "raterIndex"


class MatrixFactorization:
  def __init__(
    self,
    initLearningRate=0.2,
    noInitLearningRate=1.0,
    convergence=1e-7,
    numFactors=1,
    useGlobalIntercept=True,
    log=True,
    model: Optional[BiasedMatrixFactorization] = None,
    featureCols: List[str] = [c.noteIdKey, c.raterParticipantIdKey],
    labelCol: str = c.helpfulNumKey,
    useSigmoidCrossEntropy=False,
    posWeight=None,
    userFactorLambda=0.03,
    noteFactorLambda=0.03,
    userInterceptLambda=0.03 * 5,
    noteInterceptLambda=0.03 * 5,
    globalInterceptLambda=0.03 * 5,
    diamondLambda=0,
    normalizedLossHyperparameters=None,
    seed: Optional[int] = None,
  ) -> None:
    """Configure matrix factorization note ranking."""
    self._initLearningRate = initLearningRate
    self._noInitLearningRate = noInitLearningRate
    self._convergence = convergence
    self._numFactors = numFactors
    self._useGlobalIntercept = useGlobalIntercept
    self._log = log
    self._featureCols = featureCols
    self._labelCol = labelCol
    self._useSigmoidCrossEntropy = useSigmoidCrossEntropy
    self._posWeight = posWeight
    self._userFactorLambda = userFactorLambda
    self._noteFactorLambda = noteFactorLambda
    self._userInterceptLambda = userInterceptLambda
    self._noteInterceptLambda = noteInterceptLambda
    self._globalInterceptLambda = globalInterceptLambda
    self._diamondLambda = diamondLambda
    self._normalizedLossHyperparameters = normalizedLossHyperparameters
    self._lossModule: Optional[NormalizedLoss] = None
    self._seed = seed

    if self._useSigmoidCrossEntropy:
      if self._posWeight:
        if log:
          logger.info(f"Using pos weight: {self._posWeight} with BCEWithLogitsLoss")
        self.criterion = torch.nn.BCEWithLogitsLoss(
          pos_weight=torch.FloatTensor(np.array(self._posWeight)), reduction="none"
        )
      else:
        if log:
          logger.info("Using BCEWithLogitsLoss")
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    else:
      if self._posWeight:
        raise ValueError("posWeight is not supported for MSELoss")
      self.criterion = torch.nn.MSELoss(reduction="none")

    self.train_errors: List[float] = []
    self.test_errors: List[float] = []
    self.mf_model = model

    self.modelData: Optional[ModelData] = None
    self.trainModelData: Optional[ModelData] = None
    self.validateModelData: Optional[ModelData] = None

    self._ratingPerNoteLossRatio: Optional[float] = None
    self._ratingPerUserLossRatio: Optional[float] = None

  def get_final_train_error(self) -> Optional[float]:
    return self.train_errors[-1] if self.train_errors else None

  def get_new_mf_with_same_args(self):
    return MatrixFactorization(
      initLearningRate=self._initLearningRate,
      noInitLearningRate=self._noInitLearningRate,
      convergence=self._convergence,
      numFactors=self._numFactors,
      useGlobalIntercept=self._useGlobalIntercept,
      log=self._log,
      model=None,
      featureCols=self._featureCols,
      labelCol=self._labelCol,
      userFactorLambda=self._userFactorLambda,
      noteFactorLambda=self._noteFactorLambda,
      userInterceptLambda=self._userInterceptLambda,
      noteInterceptLambda=self._noteInterceptLambda,
      globalInterceptLambda=self._globalInterceptLambda,
    )

  def _initialize_note_and_rater_id_maps(
    self,
    ratings: pd.DataFrame,
  ) -> None:
    (
      self.noteIdMap,
      self.raterIdMap,
      self.ratingFeaturesAndLabels,
    ) = self.get_note_and_rater_id_maps(ratings)

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
    noteData = ratings[self._featureCols + [self._labelCol]]
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
      if self._log:
        logger.info("initializing notes")
      noteInit = self.noteIdMap.merge(
        noteInit,
        on=c.noteIdKey,
        how="left",
        unsafeAllowed={c.noteIdKey, "noteIndex_y"},
      )

      noteInit[c.internalNoteInterceptKey].fillna(0.0, inplace=True)
      self.mf_model.note_intercepts.weight.data = torch.tensor(
        np.expand_dims(noteInit[c.internalNoteInterceptKey].astype(np.float32).values, axis=1)
      )

      for i in range(1, self._numFactors + 1):
        noteInit[c.note_factor_key(i)].fillna(0.0, inplace=True)
      self.mf_model.note_factors.weight.data = torch.tensor(
        noteInit[[c.note_factor_key(i) for i in range(1, self._numFactors + 1)]]
        .astype(np.float32)
        .values
      )

    if userInit is not None:
      if self._log:
        logger.info("initializing users")
      userInit = self.raterIdMap.merge(userInit, on=c.raterParticipantIdKey, how="left")

      userInit[c.internalRaterInterceptKey] = userInit[c.internalRaterInterceptKey].fillna(0.0)
      self.mf_model.user_intercepts.weight.data = torch.tensor(
        np.expand_dims(userInit[c.internalRaterInterceptKey].astype(np.float32).values, axis=1)
      )

      for i in range(1, self._numFactors + 1):
        userInit[c.rater_factor_key(i)] = userInit[c.rater_factor_key(i)].fillna(0.0)
      self.mf_model.user_factors.weight.data = torch.tensor(
        userInit[[c.rater_factor_key(i) for i in range(1, self._numFactors + 1)]]
        .astype(np.float32)
        .values
      )

    if globalInterceptInit is not None:
      if self._log:
        logger.info("initialized global intercept")
      self.mf_model.global_intercept = torch.nn.parameter.Parameter(
        torch.ones(1, 1, dtype=torch.float32) * globalInterceptInit
      )

  def _get_parameters_from_trained_model(
    self, flipFactorsForIdentification: bool = True
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    for i in range(self._numFactors):
      noteParams[c.note_factor_key(i + 1)] = self.mf_model.note_factors.weight.data.cpu().numpy()[
        :, i
      ]
      raterParams[c.rater_factor_key(i + 1)] = self.mf_model.user_factors.weight.data.cpu().numpy()[
        :, i
      ]

    if flipFactorsForIdentification:
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
      logger.info(f"learning rate set to :{self._initLearningRate}")
      self.optimizer = torch.optim.Adam(
        self.mf_model.parameters(), lr=self._initLearningRate
      )  # smaller learning rate
    else:
      logger.info(f"learning rate set to :{self._noInitLearningRate}")
      self.optimizer = torch.optim.Adam(self.mf_model.parameters(), lr=self._noInitLearningRate)
    if self._log:
      logger.info(f"{self.mf_model.device}")
    self.mf_model.to(self.mf_model.device)

  def _instantiate_biased_mf_model(self):
    n_users = self.ratingFeaturesAndLabels[Constants.raterIndexKey].nunique()
    n_notes = self.ratingFeaturesAndLabels[Constants.noteIndexKey].nunique()
    self.mf_model = BiasedMatrixFactorization(
      n_users,
      n_notes,
      use_global_intercept=self._useGlobalIntercept,
      n_factors=self._numFactors,
      log=self._log,
      seed=self._seed,
    )
    if self._log:
      logger.info("------------------")
      logger.info(f"Users: {n_users}, Notes: {n_notes}")

  def _compute_and_print_loss(
    self,
    loss_value: float,
    epoch: int,
    run_name: str = "",
    final: bool = False,
  ) -> Tuple[float, float, Optional[float]]:
    assert self.mf_model is not None
    assert self.trainModelData is not None

    y_pred = self.mf_model(self.trainModelData)
    train_loss_value = self.criterion(y_pred, self.trainModelData.rating_labels).mean().item()

    metrics = {
      f"{run_name}Loss/train_fit": train_loss_value,
      f"{run_name}Loss/regularized": loss_value,
    }

    if self.validateModelData is not None:
      y_pred_validate = self.mf_model(self.validateModelData)
      validate_loss_value = (
        self.criterion(y_pred_validate, self.validateModelData.rating_labels).mean().item()
      )
      metrics[f"{run_name}Loss/validate_fit"] = validate_loss_value
    else:
      validate_loss_value = None

    if self._log:
      logger.info(f"epoch {epoch} {loss_value}")
      logger.info(f"TRAIN FIT LOSS: {train_loss_value}")
      if validate_loss_value is not None:
        logger.info(f"VALIDATE FIT LOSS: {validate_loss_value}")

    wandb.log(metrics, step=epoch, commit=False)

    if final == True:
      self.test_errors.append(loss_value)
      self.train_errors.append(train_loss_value)

      if wandb._enabled:
        # Log min, max, mean, delta, and histogram of the parameters with adaptive handling
        log_dict = {}
        for name, param in self.mf_model.named_parameters():
          param_array = param.data.cpu().clone().numpy()
          param_range = param_array.max() - param_array.min()
          key = f"{run_name}Parameters/{name}"

          log_dict[f"{key}_min"] = param_array.min()
          log_dict[f"{key}_max"] = param_array.max()
          log_dict[f"{key}_mean"] = param_array.mean()
          log_dict[f"{key}_delta"] = param_range

          if param_range > 1e-6:  # Threshold for meaningful histograms
            log_dict[key] = wandb.Histogram(sequence=param_array, num_bins=10)
          elif self._log:
            logger.info(f"Skipping histogram for {key}: range {param_range} too small")
        wandb.run.summary.update(log_dict)

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

  def _get_loss(self, epoch: Optional[int] = None):
    assert self.mf_model is not None
    y_pred = self.mf_model(self.trainModelData)
    if self._lossModule is not None:
      loss = self._lossModule(y_pred)
    else:
      assert self.trainModelData is not None
      loss = self.criterion(y_pred, self.trainModelData.rating_labels).mean()
    regularizationLoss = self._get_reg_loss()
    loss += regularizationLoss
    if torch.isnan(loss).any():
      logger.info(f"NaN loss detected at epoch {epoch}, re-initializing model")
      # Re-initialize model with stable seed
      if isinstance(self._lossModule, NormalizedLoss):
        self._create_mf_model(None, self.userInit, None)
      else:
        self._create_mf_model(self.noteInit, self.userInit, self.globalInterceptInit)
      # Compute loss with the fresh model to maintain gradient connectivity
      y_pred = self.mf_model(self.trainModelData)
      if self._lossModule is not None:
        loss = self._lossModule(y_pred)
      else:
        assert self.trainModelData is not None
        loss = self.criterion(y_pred, self.trainModelData.rating_labels).mean()
      regularizationLoss = self._get_reg_loss()
      loss += regularizationLoss
    return loss

  def _get_reg_loss(self):
    l2_reg_loss = torch.tensor(0.0, dtype=torch.float32).to(self.mf_model.device)

    if self._ratingPerUserLossRatio is None:
      l2_reg_loss += self._userFactorLambda * (self.mf_model.user_factors.weight**2).mean()
      l2_reg_loss += self._userInterceptLambda * (self.mf_model.user_intercepts.weight**2).mean()
    else:
      simulatedNumberOfRatersForLoss = (
        len(self.trainModelData.rating_labels) / self._ratingPerUserLossRatio
      )
      l2_reg_loss += (
        self._userFactorLambda
        * (self.mf_model.user_factors.weight**2).sum()
        / simulatedNumberOfRatersForLoss
      )
      l2_reg_loss += (
        self._userInterceptLambda
        * (self.mf_model.user_intercepts.weight**2).sum()
        / simulatedNumberOfRatersForLoss
      )

    if self._ratingPerNoteLossRatio is None:
      l2_reg_loss += self._noteFactorLambda * (self.mf_model.note_factors.weight**2).mean()
      l2_reg_loss += self._noteInterceptLambda * (self.mf_model.note_intercepts.weight**2).mean()
      l2_reg_loss += (
        self._diamondLambda
        * (self.mf_model.note_factors.weight * self.mf_model.note_intercepts.weight).abs().mean()
      )
    else:
      simulatedNumberOfNotesForLoss = (
        len(self.trainModelData.rating_labels) / self._ratingPerNoteLossRatio
      )
      l2_reg_loss += (
        self._noteFactorLambda
        * (self.mf_model.note_factors.weight**2).sum()
        / simulatedNumberOfNotesForLoss
      )
      l2_reg_loss += (
        self._noteInterceptLambda
        * (self.mf_model.note_intercepts.weight**2).sum()
        / simulatedNumberOfNotesForLoss
      )
      l2_reg_loss += (
        self._diamondLambda
        * (self.mf_model.note_factors.weight * self.mf_model.note_intercepts.weight).abs().sum()
        / simulatedNumberOfNotesForLoss
      )

    l2_reg_loss += self._globalInterceptLambda * (self.mf_model.global_intercept**2).mean()

    return l2_reg_loss

  def _fit_model(
    self,
    validate_percent: Optional[float] = None,
    print_interval: int = 20,
    run_name: str = "",
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
    self._create_train_validate_sets(validate_percent)
    assert self.trainModelData is not None

    prev_loss = 1e10
    loss = self._get_loss()
    epoch = 0

    while (abs(loss.item() - prev_loss) > self._convergence) and (
      not (epoch > 100 and loss.item() > prev_loss)
    ):
      prev_loss = loss.item()

      # Backpropagate
      loss.backward()
      # clip gradients to improve numerical stability
      torch.nn.utils.clip_grad_norm_(self.mf_model.parameters(), max_norm=1.0)

      # Update the parameters
      self.optimizer.step()

      # Set gradients to zero
      self.optimizer.zero_grad()

      # Predict and calculate loss
      loss = self._get_loss(epoch=epoch)

      if epoch % print_interval == 0:
        self._compute_and_print_loss(loss.item(), epoch, run_name=run_name, final=False)
        # Log histograms with adaptive handling

      epoch += 1

    if self._log:
      logger.info(f"Num epochs: {epoch}")
    return self._compute_and_print_loss(loss.item(), epoch, run_name=run_name, final=True)

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

    rating_labels = torch.FloatTensor(ratingFeaturesAndLabels[self._labelCol].values).to(
      self.mf_model.device
    )
    user_indexes = torch.IntTensor(ratingFeaturesAndLabels[Constants.raterIndexKey].values).to(
      self.mf_model.device
    )
    note_indexes = torch.IntTensor(ratingFeaturesAndLabels[Constants.noteIndexKey].values).to(
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
    freezeNoteParameters: bool = False,
    freezeRaterParameters: bool = False,
    freezeGlobalParameters: bool = False,
    ratingPerNoteLossRatio: Optional[float] = None,
    ratingPerUserLossRatio: Optional[float] = None,
    flipFactorsForIdentification: bool = True,
    run_name: str = "",
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
    if run_name:
      unique_run_id = f"{run_name}_{int(time.time())}"
      run_name += "/"
    else:
      unique_run_id = f"mf_run_{int(time.time())}"

    # Store hyperparameters in wandb config
    config = {
      "initLearningRate": self._initLearningRate,
      "noInitLearningRate": self._noInitLearningRate,
      "convergence": self._convergence,
      "numFactors": self._numFactors,
      "useGlobalIntercept": self._useGlobalIntercept,
      "useSigmoidCrossEntropy": self._useSigmoidCrossEntropy,
      "posWeight": self._posWeight,
      "userFactorLambda": self._userFactorLambda,
      "noteFactorLambda": self._noteFactorLambda,
      "userInterceptLambda": self._userInterceptLambda,
      "noteInterceptLambda": self._noteInterceptLambda,
      "globalInterceptLambda": self._globalInterceptLambda,
      "diamondLambda": self._diamondLambda,
      "ratingPerNoteLossRatio": ratingPerNoteLossRatio,
      "ratingPerUserLossRatio": ratingPerUserLossRatio,
      "freezeNoteParameters": freezeNoteParameters,
      "freezeRaterParameters": freezeRaterParameters,
      "freezeGlobalParameters": freezeGlobalParameters,
      "validatePercent": validatePercent,
      "flipFactorsForIdentification": flipFactorsForIdentification,
    }
    logger.info(f"Reinitializing wandb run with enabled state: {wandb._enabled}")
    wandb.reinitialize(unique_run_id, config=config)
    self._ratingPerNoteLossRatio = ratingPerNoteLossRatio
    self._ratingPerUserLossRatio = ratingPerUserLossRatio

    self._initialize_note_and_rater_id_maps(ratings)
    self.noteInit = noteInit
    self.userInit = userInit
    self.globalInterceptInit = globalInterceptInit
    self._create_mf_model(noteInit, userInit, globalInterceptInit)
    assert self.mf_model is not None

    logger.info(
      f"Ratings per note in dataset: {len(ratings)/self.mf_model.note_factors.weight.data.shape[0]}"
    )
    logger.info(
      f"Ratings per user in dataset: {len(ratings)/self.mf_model.user_factors.weight.data.shape[0]}"
    )
    if ratingPerNoteLossRatio is not None:
      logger.info(
        f"Correcting loss function to simulate rating per note loss ratio = {ratingPerNoteLossRatio}"
      )
    if ratingPerUserLossRatio is not None:
      logger.info(
        f"Correcting loss function to simulate rating per user loss ratio = {ratingPerUserLossRatio}"
      )

    if freezeRaterParameters:
      self.mf_model._freeze_parameters(set({"user"}))
    if freezeGlobalParameters:
      self.mf_model._freeze_parameters(set({"global"}))
    if freezeNoteParameters:
      self.mf_model._freeze_parameters(set({"note"}))
    if specificNoteId is not None:
      self.mf_model.freeze_rater_and_global_parameters()
    self.prepare_features_and_labels(specificNoteId)

    train_loss, loss, validate_loss = self._fit_model(validatePercent, run_name=run_name)
    if self._normalizedLossHyperparameters is not None:
      _, raterParams = self._get_parameters_from_trained_model(flipFactorsForIdentification)
      assert self.modelData is not None
      self._lossModule = NormalizedLoss(
        self.criterion,
        self.ratingFeaturesAndLabels,
        self.modelData.rating_labels,
        self._normalizedLossHyperparameters,
        self._labelCol,
        raterParams,
        device=self.mf_model.device,
      )
      self._create_mf_model(None, userInit, None)
      train_loss, loss, validate_loss = self._fit_model(validatePercent, run_name=run_name)
      self._lossModule = None

    assert self.mf_model.note_factors.weight.data.cpu().numpy().shape[0] == self.noteIdMap.shape[0]

    globalIntercept = None
    if self._useGlobalIntercept:
      globalIntercept = self.mf_model.global_intercept.item()
      if self._log:
        logger.info(f"Global Intercept: {globalIntercept}")

    fitNoteParams, fitRaterParams = self._get_parameters_from_trained_model(
      flipFactorsForIdentification
    )

    wandb.finish()

    fitRaterParams.drop(Constants.raterIndexKey, axis=1, inplace=True)
    if validatePercent is None:
      return fitNoteParams, fitRaterParams, globalIntercept
    else:
      return (
        fitNoteParams,
        fitRaterParams,
        globalIntercept,
        train_loss,
        loss,
        validate_loss,
      )

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
    for i in range(1, self._numFactors + 1):
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
