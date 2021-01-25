---
title: Note ranking
aliases:
  [
    "/ranking-notes",
    "/note-ranking",
    "/about/note-ranking",
    "/about/ranking-notes",
  ]
geekdocBreadcrumb: false
---

# Note ranking

Birdwatch initially shows notes in reverse-chronological order (most recent on top). In some cases — if notes have received enough ratings from other contributors, and certain criteria are met — notes might be raised to the top of the list and highlighted with a header showing that they are **“currently rated helpful.”**

To help people understand how Birdwatch works, this page and code snippet below explain how notes are identified as “currently rated helpful.”

{{< figure src="../../images/5-good-notes.png">}}

In short, Birdwatch computes a “smoothed helpfulness score” for each note, which is simply the proportion of ratings that say the note is helpful, smoothed by adding 5 to the denominator. Then out of all notes that have received at least three ratings, the top 3 by “smoothed helpfulness score” are annotated as “currently rated helpful” as long as the ”smoothed helpfulness score” is at least 0.5.

This ranking mechanism is knowingly basic, and we only intend to use it for a short time during the program’s earliest phase.

During the pilot, “currently rated helpful” notes are only computed at periodic intervals, so there is a time delay from when a note meets the “currently rated helpful” criteria and when it rises to the top of the list. This delay allows Birdwatch to collect a set of independent ratings from people who haven’t yet been influenced by seeing the “currently rated helpful” flag on certain notes.

Here’s a Python code snippet one can run to reproduce how we compute our “Currently Rated Helpful” annotations. It uses as input the notes and ratings files made available on the Birdwatch [Data Download](https://twitter.com/i/birdwatch/download-data) page.

{{< highlight python >}}

import pandas as pd
notes = pd.read_csv('notes-00000.tsv', sep='\t')
ratings = pd.read_csv('ratings-00000.tsv', sep='\t')

ratingsWithNotes = notes.set_index('noteId').join(ratings.set_index('noteId'), lsuffix="\_note", rsuffix="\_rating", how='inner')
ratingsWithNotes['numRatings'] = 1

def getCurrentlyRatedHelpfulNotesForTweet(
  tweetId,
  noteScoreSmoothingParameter = 5,
  maxCurrentlyRatedHelpfulNotes = 3,
  minRatingsNeeded = 3,
  minSmoothedHelpfulnessScoreNeeded = 0.5
):
  ratingsWithNotesForTweet = ratingsWithNotes[ratingsWithNotes['tweetId']==tweetId]
  scoredNotes = ratingsWithNotesForTweet.groupby('noteId').sum()
  scoredNotes['smoothedHelpfulnessScore'] = scoredNotes['helpful']/(scoredNotes['numRatings'] + noteScoreSmoothingParameter)
  filteredNotes = scoredNotes[(scoredNotes['numRatings'] >= minRatingsNeeded) & (scoredNotes['smoothedHelpfulnessScore'] >= minSmoothedHelpfulnessScoreNeeded)]
  return filteredNotes.sort_values(by='smoothedHelpfulnessScore', ascending=False)[:maxCurrentlyRatedHelpfulNotes]

{{< / highlight >}}
