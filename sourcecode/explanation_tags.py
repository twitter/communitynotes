import numpy as np
import pandas as pd
from constants import *

helpfulTags = [
  "helpfulOther",
  "helpfulUnbiasedLanguage",
  "helpfulUniqueContext",
  "helpfulEmpathetic",
  "helpfulGoodSources",
  "helpfulAddressesClaim",
  "helpfulImportantContext",
  "helpfulClear",
  "helpfulInformative",
]

nothelpfulTags = [
  "notHelpfulOther",
  "notHelpfulIrrelevantSources",
  "notHelpfulSourcesMissingOrUnreliable",
  "notHelpfulOpinionSpeculationOrBias",
  "notHelpfulMissingKeyPoints",
  "notHelpfulNoteNotNeeded",
  "notHelpfulArgumentativeOrBiased",
  "notHelpfulIncorrect",
  "notHelpfulOffTopic",
  "notHelpfulHardToUnderstand",
  "notHelpfulSpamHarassmentOrAbuse",
  "notHelpfulOutdated",
]

def top_tags(row, minRatingsToGetTag, minTagsNeededForStatus):
  if row[ratingStatusKey] == currentlyRatedHelpful:
    tagCounts = pd.DataFrame(row[helpfulTags])
  elif row[ratingStatusKey] == currentlyRatedNotHelpful:
    tagCounts = pd.DataFrame(row[nothelpfulTags])
  else:
    return row
  tagCounts.columns = [tagCountsKey]
  tagCounts[tiebreakOrderKey] = range(len(tagCounts))
  tagCounts = tagCounts[tagCounts[tagCountsKey] >= minRatingsToGetTag]
  topTags = tagCounts.sort_values(by=[tagCountsKey, tiebreakOrderKey], ascending=False)[:2]
  if len(topTags) < minTagsNeededForStatus:
    row[ratingStatusKey] = needsMoreRatings
  else:
    row[firstTagKey] = topTags.index[0]
    row[secondTagKey] = topTags.index[1]
  return row


def get_rating_status_and_explanation_tags(
  notes,
  ratings,
  noteParams,
  minRatingsNeeded,
  minRatingsToGetTag,
  minTagsNeededForStatus,
  crhThreshold,
  crnhThreshold,
  logging=True,
):
  ratingsWithNotes = notes.set_index(noteIdKey).join(
    ratings.set_index(noteIdKey), lsuffix="\_note", rsuffix="\_rating", how="inner"
  )
  ratingsWithNotes[numRatingsKey] = 1
  scoredNotes = ratingsWithNotes.groupby(noteIdKey).sum()
  scoredNotes = scoredNotes.merge(noteParams[[noteIdKey, noteInterceptKey]], on=noteIdKey)

  scoredNotes[ratingStatusKey] = needsMoreRatings
  scoredNotes.loc[
    (scoredNotes[numRatingsKey] >= minRatingsNeeded)
    & (scoredNotes[noteInterceptKey] >= crhThreshold),
    ratingStatusKey,
  ] = currentlyRatedHelpful
  scoredNotes.loc[
    (scoredNotes[numRatingsKey] >= minRatingsNeeded)
    & (scoredNotes[noteInterceptKey] <= crnhThreshold),
    ratingStatusKey,
  ] = currentlyRatedNotHelpful

  scoredNotes[firstTagKey] = np.nan
  scoredNotes[secondTagKey] = np.nan

  scoredNotes = scoredNotes.apply(
    lambda row: top_tags(row, minRatingsToGetTag, minTagsNeededForStatus), axis=1
  )

  if logging:
    print("Num Scored Notes:", len(scoredNotes))
  return scoredNotes
