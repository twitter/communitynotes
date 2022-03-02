import explanation_tags
import helpfulness_scores
import matrix_factorization
import process_data
from constants import *



ratings, notes = process_data.get_data()
ratingsForTraining = process_data.filter_ratings(ratings)

noteParamsUnfiltered, raterParamsUnfiltered, globalBias = matrix_factorization.run_mf(
  ratingsForTraining,
  l2_lambda,
  l2_intercept_multiplier,
  numFactors,
  epochs,
  useGlobalIntercept,
  runName="unfiltered",
)

helpfulnessScores = helpfulness_scores.compute_general_helpfulness_scores(
  noteParamsUnfiltered,
  notes,
  ratings,
  crhThreshold,
  crnhThreshold,
  minRatingsNeeded,
)

ratingsHelpfulnessScoreFiltered = helpfulness_scores.filter_ratings_by_helpfulness_scores(
  ratingsForTraining, helpfulnessScores, minMeanNoteScore, minCRHVsCRNHRatio, minRaterAgreeRatio
)

noteParams, raterParams, globalBias = matrix_factorization.run_mf(
  ratingsHelpfulnessScoreFiltered,
  l2_lambda,
  l2_intercept_multiplier,
  numFactors,
  epochs,
  useGlobalIntercept,
)

scoredNotes = explanation_tags.get_rating_status_and_explanation_tags(
  notes,
  ratings,
  noteParams,
  minRatingsNeeded,
  minRatingsToGetTag,
  minTagsNeededForStatus,
  crhThreshold,
  crnhThreshold,
)

scoredNotes = scoredNotes.set_index(noteIdKey).join(
  notes[[noteIdKey, "summary"]].set_index(noteIdKey), lsuffix="_note", rsuffix="_rating", how="inner"
)
scoredNotes.to_csv("scoredNotes.tsv", sep="\t", index=False)
