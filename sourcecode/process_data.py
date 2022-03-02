import numpy as np
import pandas as pd
from constants import *

def get_data(logging=True):
  notes = pd.read_csv(notesInputPath, sep="\t")
  ratings = pd.read_csv(ratingsInputPath, sep="\t")

  ratings = ratings.rename({participantIdKey: raterParticipantIdKey}, axis=1)
  ratings[helpfulNumKey] = np.nan
  ratings.loc[ratings[helpfulKey] == 1, helpfulNumKey] = 1
  ratings.loc[ratings[notHelpfulKey] == 1, helpfulNumKey] = 0
  ratings.loc[ratings[helpfulnessLevelKey] == notHelpfulValueTsv, helpfulNumKey] = 0
  ratings.loc[ratings[helpfulnessLevelKey] == somewhatHelpfulValueTsv, helpfulNumKey] = 0.5
  ratings.loc[ratings[helpfulnessLevelKey] == helpfulValueTsv, helpfulNumKey] = 1
  ratings = ratings.loc[~pd.isna(ratings[helpfulNumKey])]

  if logging:
    print(
      "Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (len(ratings), len(np.unique(ratings[noteIdKey])), len(np.unique(ratings[raterParticipantIdKey])))
    )
  return ratings, notes


## Apply min # of ratings for raters & notes.
def filter_ratings(ratings, logging=True):
  ratingsTriplets = ratings[[raterParticipantIdKey, noteIdKey, helpfulNumKey, createdAtMillisKey]]
  n = ratingsTriplets.groupby(noteIdKey).size().reset_index()
  notesWithMinNumRatings = n[n[0] >= minNumRatersPerNote]

  ratingsNoteFiltered = ratingsTriplets.merge(notesWithMinNumRatings[[noteIdKey]], on=noteIdKey)
  if logging:
    print(
      "After Note Filtering: Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        len(ratingsNoteFiltered),
        len(np.unique(ratingsNoteFiltered[noteIdKey])),
        len(np.unique(ratingsNoteFiltered[raterParticipantIdKey])),
      )
    )
  r = ratingsNoteFiltered.groupby(raterParticipantIdKey).size().reset_index()
  ratersWithMinNumRatings = r[r[0] >= 10]

  ratingsDoubleFiltered = ratingsNoteFiltered.merge(ratersWithMinNumRatings[[raterParticipantIdKey]], on=raterParticipantIdKey)
  if logging:
    print(
      "After Rating Filtering: Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        len(ratingsDoubleFiltered),
        len(np.unique(ratingsDoubleFiltered[noteIdKey])),
        len(np.unique(ratingsDoubleFiltered[raterParticipantIdKey])),
      )
    )
  n = ratingsDoubleFiltered.groupby(noteIdKey).size().reset_index()
  notesWithMinNumRatings = n[n[0] >= minNumRatersPerNote]
  ratingsForTraining = ratingsDoubleFiltered.merge(notesWithMinNumRatings[[noteIdKey]], on=noteIdKey)
  if logging:
    print(
      "After Final Filtering: Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        len(ratingsForTraining),
        len(np.unique(ratingsForTraining[noteIdKey])),
        len(np.unique(ratingsForTraining[raterParticipantIdKey])),
      )
    )

  ratingsForTraining = ratingsForTraining
  return ratingsForTraining
