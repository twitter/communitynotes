import logging
from typing import Optional

from . import constants as c, process_data
from .matrix_factorization.matrix_factorization import MatrixFactorization

import numpy as np
import pandas as pd


logger = logging.getLogger("birdwatch.tag_consensus")
logger.setLevel(logging.INFO)


def train_tag_model(
  ratings: pd.DataFrame,
  tag: str = c.notHelpfulSpamHarassmentOrAbuseTagKey,
  helpfulModelNoteParams: pd.DataFrame = None,
  helpfulModelRaterParams: pd.DataFrame = None,
  useSigmoidCrossEntropy: bool = True,
  name: Optional[str] = None,
):
  def _convert_param_name_from_internal_to_tag(
    paramName: str,
    name: Optional[str] = name,
    tag: str = tag,
  ) -> str:
    if name is None:
      name = tag.split("elpful")[-1]
    return paramName.replace("internal", name)

  # Prep empty dataframes to return in case we abort training.
  emptyNoteParams = helpfulModelNoteParams[[c.noteIdKey]]
  emptyNoteParams[
    [
      _convert_param_name_from_internal_to_tag(c.internalNoteInterceptKey),
      _convert_param_name_from_internal_to_tag(c.internalNoteFactor1Key),
    ]
  ] = np.nan
  emptyRaterParams = helpfulModelRaterParams[[c.raterParticipantIdKey]]
  emptyRaterParams[
    [
      _convert_param_name_from_internal_to_tag(c.internalRaterInterceptKey),
      _convert_param_name_from_internal_to_tag(c.internalRaterFactor1Key),
    ]
  ] = np.nan

  logger.info(f"-------------------Training for tag {tag}-------------------")
  ratingDataForTag, labelColName = prepare_tag_data(ratings, tag)
  if ratingDataForTag is None or len(ratingDataForTag) == 0:
    logger.info(f"No valid data for {tag}, returning None and aborting {tag} model training.")
    return emptyNoteParams, emptyRaterParams, None

  posRate = ratingDataForTag[labelColName].sum() / len(ratingDataForTag)
  logger.info(f"{tag} Positive Rate: {posRate}")
  if pd.isna(posRate) or posRate == 0 or posRate == 1:
    logger.info(
      f"{tag} tag positive rate is {posRate}: returning None and aborting {tag} model training."
    )
    return emptyNoteParams, emptyRaterParams, None

  if useSigmoidCrossEntropy:
    posWeight = (1 - posRate) / posRate
  else:
    posWeight = None

  # Train
  mf = MatrixFactorization(
    labelCol=labelColName,
    useSigmoidCrossEntropy=useSigmoidCrossEntropy,
    posWeight=posWeight,
    initLearningRate=2.0,
  )

  # Initialize model with note and user factors from the helpfulness model, to improve stability.
  # But set intercepts to 0, since it's a different outcome variable than helpfulness.
  if helpfulModelNoteParams is not None:
    assert c.internalNoteInterceptKey in helpfulModelNoteParams.columns
    helpfulModelNoteParams = helpfulModelNoteParams.copy()
    helpfulModelNoteParams[c.internalNoteInterceptKey] = 0.0
    helpfulModelNoteParams[c.internalNoteFactor1Key] = (
      2.0 * helpfulModelNoteParams[c.internalNoteFactor1Key]
    )
  if helpfulModelRaterParams is not None:
    assert c.internalRaterInterceptKey in helpfulModelRaterParams.columns
    helpfulModelRaterParams = helpfulModelRaterParams.copy()
    helpfulModelRaterParams[c.internalRaterInterceptKey] = 0.0
    helpfulModelRaterParams[c.internalRaterFactor1Key] = (
      2.0 * helpfulModelRaterParams[c.internalRaterFactor1Key]
    )

  noteParams, raterParams, globalBias = mf.run_mf(
    ratingDataForTag,
    userInit=helpfulModelRaterParams,
    noteInit=helpfulModelNoteParams,
    run_name="tag_consensus",
  )

  noteParams.columns = [_convert_param_name_from_internal_to_tag(col) for col in noteParams.columns]
  raterParams.columns = [
    _convert_param_name_from_internal_to_tag(col) for col in raterParams.columns
  ]
  return noteParams, raterParams, globalBias


def prepare_tag_data(
  allRatings: pd.DataFrame,
  tagName: str = c.notHelpfulIncorrectTagKey,
  minNumRatingsPerRater: int = 10,
  minNumRatersPerNote: int = 5,
):
  ratings = allRatings.loc[
    allRatings[c.createdAtMillisKey] >= c.lastRatingTagsChangeTimeMillis
  ].copy()
  if len(ratings) == 0:
    return None, None

  labelColName = tagName + "Label"
  ratings.loc[:, labelColName] = None

  if tagName.startswith("helpful"):
    oppositeValenceHelpfulness = c.notHelpfulValueTsv
    sameValenceOtherTag = c.helpfulOtherTagKey
  elif tagName.startswith("notHelpful"):
    oppositeValenceHelpfulness = c.helpfulValueTsv
    sameValenceOtherTag = c.notHelpfulOtherTagKey
  else:
    raise Exception("Tag unsupported.")

  # Negatives: opposite helpful rating, or same-valence rating with other tag (and no target tag)
  ratings.loc[ratings[c.helpfulnessLevelKey] == oppositeValenceHelpfulness, labelColName] = 0
  # Treat a same-valence rating as negative if the tag used was other
  #  (other is the only tag uncorrelated enough with other tags))
  ## Will be set to True later if the tag itself was also true
  ratings.loc[ratings[sameValenceOtherTag] == 1, labelColName] = 0

  # Positives
  ratings.loc[ratings[tagName] == 1, labelColName] = 1

  logger.info(f"Pre-filtering tag label breakdown {ratings.groupby(labelColName).size()}")
  logger.info(f"Number of rows with no tag label {ratings[labelColName].isnull().sum()}")

  # Currently leave in raters who only made one type of rating, but can throw them out in the future.
  ratings = process_data.filter_ratings(
    ratings[ratings[labelColName].notnull()], minNumRatingsPerRater, minNumRatersPerNote
  )

  logger.info(f"Post-filtering tag label breakdown {ratings.groupby(labelColName).size()}")
  logger.info(f"Number of rows with no tag label {ratings[labelColName].isnull().sum()}")

  ratings[labelColName] = ratings[labelColName].astype(int)

  return ratings, labelColName
