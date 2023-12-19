from dataclasses import dataclass

from .. import constants as c
from .matrix_factorization import Constants as mf_c, MatrixFactorization

import pandas as pd
import torch


@dataclass
class Constants:
  extraRaterInterceptKey = "extraRaterIntercept"
  extraRaterFactor1Key = "extraRaterFactor1"
  extraRatingHelpfulNumKey = "extraRatingHelpfulNum"
  originalKey = "original"
  refitOriginalKey = "refit_orig"
  allKey = "all"
  negFacKey = "neg_fac"
  posFacKey = "pos_fac"


class PseudoRatersRunner:
  def __init__(
    self,
    ratings: pd.DataFrame,
    noteParams: pd.DataFrame,
    raterParams: pd.DataFrame,
    globalBias: float,
    mfRanker: MatrixFactorization,
    logging=True,
    checkParamsSame=True,
  ):
    self._logging = logging
    self._mfRanker = mfRanker
    self._checkParamsSame = checkParamsSame
    self.ratings = ratings
    (
      self.noteIdMap,
      self.raterIdMap,
      self.ratingFeaturesAndLabels,
    ) = self._mfRanker.get_note_and_rater_id_maps(ratings)
    self.noteParams = noteParams
    self.raterParams = raterParams
    self.globalBias = globalBias

  def compute_note_parameter_confidence_bounds_with_pseudo_raters(self):
    with c.time_block("Pseudoraters: prepare data"):
      self._make_extreme_raters(self.raterParams, self.raterIdMap)
      self._add_extreme_raters_to_id_maps_and_params()
      self._create_extreme_ratings()

    with c.time_block("Pseudoraters: fit models"):
      noteParamsList = self._fit_note_params_for_each_dataset_with_extreme_ratings()

    with c.time_block("Pseudoraters: aggregate"):
      notesWithConfidenceBounds = self._aggregate_note_params(noteParamsList)
      noteParams = self.noteParams.merge(
        notesWithConfidenceBounds.reset_index(), on=c.noteIdKey, how="left"
      )
    return noteParams

  def _check_rater_parameters_same(self, newMatrixFactorization: MatrixFactorization):
    (
      noteParamsFromNewModel,
      raterParamsFromNewModel,
    ) = newMatrixFactorization._get_parameters_from_trained_model()
    raterParamsCols = [
      c.raterParticipantIdKey,
      c.internalRaterInterceptKey,
      c.internalRaterFactor1Key,
    ]
    newParams = raterParamsFromNewModel.loc[:, raterParamsCols].set_index(c.raterParticipantIdKey)
    oldParams = self.raterParams.loc[:, raterParamsCols].set_index(c.raterParticipantIdKey)
    overlapParticipantIds = newParams.index.intersection(oldParams.index)
    assert len(overlapParticipantIds) == len(oldParams)
    assert (
      (newParams.loc[overlapParticipantIds] == oldParams.loc[overlapParticipantIds]).all().all()
    )

  def _check_note_parameters_same(self, newMatrixFactorization: MatrixFactorization):
    (
      noteParamsFromNewModel,
      raterParamsFromNewModel,
    ) = newMatrixFactorization._get_parameters_from_trained_model()
    assert (noteParamsFromNewModel == self.noteParams).all().all()

  def _make_extreme_raters(self, raterParams: pd.DataFrame, raterIdMap: pd.DataFrame):
    """Populates self.extremeRaters, which is a list of dicts with rater id info

    Args:
        raterParams (_type_): _description_
        raterIdMap (_type_): _description_
    """
    raterInterceptValues = [
      raterParams[c.internalRaterInterceptKey].min(),
      raterParams[c.internalRaterInterceptKey].max(),
    ]
    raterFactorValues = [
      raterParams[c.internalRaterFactor1Key].min(),
      0.0,
      raterParams[c.internalRaterFactor1Key].max(),
    ]

    self.extremeRaters = []
    i = 0
    for raterIntercept in raterInterceptValues:
      for raterFactor in raterFactorValues:
        # These pseudo-raters need to have IDs that don't conflict with real raterParticipantIds
        raterParticipantId = -1 - i
        raterIndex = raterIdMap[mf_c.raterIndexKey].max() + 1 + i

        self.extremeRaters.append(
          {
            c.raterParticipantIdKey: raterParticipantId,
            mf_c.raterIndexKey: raterIndex,
            c.internalRaterInterceptKey: raterIntercept,
            c.internalRaterFactor1Key: raterFactor,
          }
        )
        i += 1

  def _add_extreme_raters_to_id_maps_and_params(self):
    """Adds extreme raters to a new copy of raterIdMap and raterParams called
    self.raterIdMapWithExtreme and self.raterParamsWithExtreme
    """
    # TODO: should concat all at once for efficiency instead of iterative appends
    self.raterIdMapWithExtreme = self.raterIdMap.copy(deep=True)
    self.raterParamsWithExtreme = self.raterParams.copy(deep=True)

    for i, raterDict in enumerate(self.extremeRaters):
      if not (self.raterIdMap[c.raterParticipantIdKey] == raterDict[c.raterParticipantIdKey]).any():
        self.raterIdMapWithExtreme = pd.concat(
          [
            self.raterIdMapWithExtreme,
            pd.DataFrame(
              {
                c.raterParticipantIdKey: [raterDict[c.raterParticipantIdKey]],
                mf_c.raterIndexKey: [raterDict[mf_c.raterIndexKey]],
              }
            ),
          ]
        )

      if not (
        self.raterParams[c.raterParticipantIdKey] == raterDict[c.raterParticipantIdKey]
      ).any():
        self.raterParamsWithExtreme = pd.concat(
          [
            self.raterParamsWithExtreme,
            pd.DataFrame(
              {
                c.raterParticipantIdKey: [raterDict[c.raterParticipantIdKey]],
                c.internalRaterInterceptKey: [raterDict[c.internalRaterInterceptKey]],
                c.internalRaterFactor1Key: [raterDict[c.internalRaterFactor1Key]],
              }
            ),
          ]
        )

  def _create_new_model_with_extreme_raters_from_original_params(
    self, ratingFeaturesAndLabelsWithExtremeRatings: pd.DataFrame
  ):
    """Create a new instance of BiasedMatrixFactorization, initialized with the original
    model parameters, but with extreme raters added to the raterIdMap and raterParams
    """
    newExtremeMF = self._mfRanker.get_new_mf_with_same_args()
    newExtremeMF.noteIdMap = self.noteIdMap
    newExtremeMF.raterIdMap = self.raterIdMapWithExtreme
    newExtremeMF.ratingFeaturesAndLabels = ratingFeaturesAndLabelsWithExtremeRatings

    newExtremeMF._create_mf_model(
      self.noteParams.copy(deep=True), self.raterParamsWithExtreme.copy(deep=True), self.globalBias
    )
    newExtremeMF.mf_model.freeze_rater_and_global_parameters()

    newExtremeMF.optimizer = torch.optim.Adam(
      list(newExtremeMF.mf_model.note_intercepts.parameters())
      + list(newExtremeMF.mf_model.note_factors.parameters()),
      lr=newExtremeMF._initLearningRate,
    )

    if self._checkParamsSame:
      self._check_note_parameters_same(newExtremeMF)
      self._check_rater_parameters_same(newExtremeMF)

    return newExtremeMF

  def _fit_all_notes_with_raters_constant(self, ratingFeaturesAndLabelsWithExtremeRatings):
    newExtremeMF = self._create_new_model_with_extreme_raters_from_original_params(
      ratingFeaturesAndLabelsWithExtremeRatings
    )
    newExtremeMF.prepare_features_and_labels()
    newExtremeMF._fit_model()

    # Double check that we kept rater parameters fixed during re-training of note parameters.
    if self._checkParamsSame:
      self._check_rater_parameters_same(newExtremeMF)

    fitNoteParams, fitRaterParams = newExtremeMF._get_parameters_from_trained_model()
    return fitNoteParams

  def _create_extreme_ratings(self):
    self.extremeRatingsToAddWithoutNotes = []
    self.extremeRatingsToAddWithoutNotes.append(
      {
        c.internalRaterInterceptKey: None,
        c.internalRaterFactor1Key: None,
        c.helpfulNumKey: None,
      }
    )
    for extremeRater in self.extremeRaters:
      extremeRater[c.raterParticipantIdKey] = str(extremeRater[c.raterParticipantIdKey])

      for helpfulNum in (0.0, 1.0):
        extremeRater[c.helpfulNumKey] = helpfulNum
        self.extremeRatingsToAddWithoutNotes.append(extremeRater.copy())

  def _create_dataset_with_extreme_rating_on_each_note(self, ratingToAddWithoutNoteId):
    ## for each rating (ided by raterParticipantId and raterIndex)
    if ratingToAddWithoutNoteId[c.helpfulNumKey] is not None:
      ratingsWithNoteIds = []
      for noteRow in (
        self.ratingFeaturesAndLabels[[c.noteIdKey, mf_c.noteIndexKey]]
        .drop_duplicates()
        .itertuples()
      ):
        ratingToAdd = ratingToAddWithoutNoteId.copy()
        ratingToAdd[c.noteIdKey] = getattr(noteRow, c.noteIdKey)
        ratingToAdd[mf_c.noteIndexKey] = getattr(noteRow, mf_c.noteIndexKey)
        ratingsWithNoteIds.append(ratingToAdd)
      extremeRatingsToAdd = pd.DataFrame(ratingsWithNoteIds).drop(
        [c.internalRaterInterceptKey, c.internalRaterFactor1Key], axis=1
      )
      ratingFeaturesAndLabelsWithExtremeRatings = pd.concat(
        [self.ratingFeaturesAndLabels, extremeRatingsToAdd]
      )
    else:
      ratingFeaturesAndLabelsWithExtremeRatings = self.ratingFeaturesAndLabels
    return ratingFeaturesAndLabelsWithExtremeRatings

  def _fit_note_params_for_each_dataset_with_extreme_ratings(self):
    noteParamsList = []
    for ratingToAddWithoutNoteId in self.extremeRatingsToAddWithoutNotes:
      ratingFeaturesAndLabelsWithExtremeRatings = (
        self._create_dataset_with_extreme_rating_on_each_note(ratingToAddWithoutNoteId)
      )

      if self._logging:
        print("------------------")
        print(f"Re-scoring all notes with extra rating added: {ratingToAddWithoutNoteId}")

      with c.time_block("Pseudo: fit all notes with raters constant"):
        fitNoteParams = self._fit_all_notes_with_raters_constant(
          ratingFeaturesAndLabelsWithExtremeRatings
        )

      fitNoteParams[Constants.extraRaterInterceptKey] = ratingToAddWithoutNoteId[
        c.internalRaterInterceptKey
      ]
      fitNoteParams[Constants.extraRaterFactor1Key] = ratingToAddWithoutNoteId[
        c.internalRaterFactor1Key
      ]
      fitNoteParams[Constants.extraRatingHelpfulNumKey] = ratingToAddWithoutNoteId[c.helpfulNumKey]
      noteParamsList.append(fitNoteParams)
    return noteParamsList

  def _aggregate_note_params(self, noteParamsList, joinOrig=False):
    rawRescoredNotesWithEachExtraRater = pd.concat(noteParamsList)
    rawRescoredNotesWithEachExtraRater.drop(mf_c.noteIndexKey, axis=1, inplace=True)
    rawRescoredNotesWithEachExtraRater = rawRescoredNotesWithEachExtraRater.sort_values(
      by=[c.noteIdKey, Constants.extraRaterInterceptKey]
    )

    rawRescoredNotesWithEachExtraRaterAgg = (
      rawRescoredNotesWithEachExtraRater[
        [c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]
      ]
      .groupby(c.noteIdKey)
      .agg({"min", "median", "max"})
    )

    refitSameRatings = rawRescoredNotesWithEachExtraRater[
      pd.isna(rawRescoredNotesWithEachExtraRater[Constants.extraRaterInterceptKey])
    ][[c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]].set_index(c.noteIdKey)
    refitSameRatings.columns = pd.MultiIndex.from_product(
      [refitSameRatings.columns, [Constants.refitOriginalKey]]
    )
    notesWithConfidenceBounds = refitSameRatings.join(rawRescoredNotesWithEachExtraRaterAgg)

    if joinOrig:
      orig = self.noteParams[
        [c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]
      ].set_index(c.noteIdKey)
      orig.columns = pd.MultiIndex.from_product([orig.columns, [Constants.originalKey]])
      notesWithConfidenceBounds = notesWithConfidenceBounds.join(orig)

    raterFacs = self.ratingFeaturesAndLabels.merge(self.raterParams, on=c.raterParticipantIdKey)
    raterFacs[Constants.allKey] = 1
    raterFacs[Constants.negFacKey] = raterFacs[c.internalRaterFactor1Key] < 0
    raterFacs[Constants.posFacKey] = raterFacs[c.internalRaterFactor1Key] > 0
    r = raterFacs.groupby(c.noteIdKey)[
      [Constants.allKey, Constants.negFacKey, Constants.posFacKey]
    ].sum()
    r.columns = pd.MultiIndex.from_product([[c.ratingCountKey], r.columns])
    notesWithConfidenceBounds = notesWithConfidenceBounds.join(r)

    def flatten_column_names(c):
      if type(c) == tuple:
        return f"{c[0]}_{c[1]}"
      else:
        return c

    notesWithConfidenceBounds.columns = [
      flatten_column_names(c) for c in notesWithConfidenceBounds.columns
    ]
    notesWithConfidenceBounds = notesWithConfidenceBounds[
      notesWithConfidenceBounds.columns.sort_values()
    ]

    return notesWithConfidenceBounds
